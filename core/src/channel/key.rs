use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::{
    channel_sf::ChannelSoundfont, event::KeyNoteEvent, voice_buffer::VoiceBuffer,
    ChannelInitOptions, VoiceControlData,
};

/// Amplitude threshold below which voices are considered silent
const SILENCE_THRESHOLD: f32 = 0.001;

pub struct KeyData {
    key: u8,
    voices: VoiceBuffer,
    last_voice_count: usize,
    shared_voice_counter: Arc<AtomicU64>,
    max_voices_per_frame: usize,
}

impl KeyData {
    pub fn new(
        key: u8,
        shared_voice_counter: Arc<AtomicU64>,
        options: ChannelInitOptions,
    ) -> KeyData {
        KeyData {
            key,
            voices: VoiceBuffer::new(options),
            last_voice_count: 0,
            shared_voice_counter,
            max_voices_per_frame: options.max_voices_per_frame.max(1),
        }
    }

    pub fn send_event(
        &mut self,
        event: KeyNoteEvent,
        control: &VoiceControlData,
        channel_sf: &ChannelSoundfont,
        max_layers: Option<usize>,
    ) {
        match event {
            KeyNoteEvent::On(vel) => {
                let voices = channel_sf.spawn_voices_attack(control, self.key, vel);
                self.voices.push_voices(voices, max_layers);
            }
            KeyNoteEvent::Off => {
                let vel = self.voices.release_next_voice();
                if let Some(vel) = vel {
                    let voices = channel_sf.spawn_voices_release(control, self.key, vel);
                    self.voices.push_voices(voices, max_layers);
                }
            }
            KeyNoteEvent::AllOff => {
                while let Some(vel) = self.voices.release_next_voice() {
                    let voices = channel_sf.spawn_voices_release(control, self.key, vel);
                    self.voices.push_voices(voices, max_layers);
                }
            }
            KeyNoteEvent::AllKilled => {
                self.voices.kill_all_voices();
            }
        }
    }

    pub fn process_controls(&mut self, control: &VoiceControlData) {
        for voice in &mut self.voices.iter_voices_mut() {
            voice.process_controls(control);
        }
    }

    /// Render voices to output buffer with adaptive quality
    /// When voice count is high, only render the loudest voices
    pub fn render_to(&mut self, out: &mut [f32]) {
        let voice_count = self.voices.voice_count();
        
        if voice_count == 0 {
            self.update_voice_counter(0);
            return;
        }

        // Fast path: small number of voices, render all
        if voice_count <= self.max_voices_per_frame {
            for voice in &mut self.voices.iter_voices_mut() {
                voice.render_to(out);
            }
        } else {
            // Slow path: many voices, sort by amplitude and render only the loudest
            self.render_with_priority(out);
        }

        self.voices.remove_ended_voices();
        self.update_voice_counter(self.voices.voice_count());
    }

    /// Render only the highest amplitude voices when overloaded
    fn render_with_priority(&mut self, out: &mut [f32]) {
        // Collect voice indices and amplitudes for sorting
        let mut voice_data: Vec<(usize, f32)> = self.voices
            .iter_voices_mut()
            .enumerate()
            .map(|(idx, voice)| (idx, voice.amplitude()))
            .filter(|(_, amp)| *amp > SILENCE_THRESHOLD)
            .collect();

        if voice_data.is_empty() {
            return;
        }

        // Sort by amplitude descending (highest first)
        voice_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Render only top N voices
        let render_count = voice_data.len().min(self.max_voices_per_frame);
        let indices_to_render: Vec<usize> = voice_data[..render_count]
            .iter()
            .map(|(idx, _)| *idx)
            .collect();

        // Render selected voices
        for (current_idx, voice) in self.voices.iter_voices_mut().enumerate() {
            if indices_to_render.contains(&current_idx) {
                voice.render_to(out);
            }
        }
    }

    #[inline(always)]
    fn update_voice_counter(&mut self, new_count: usize) {
        let change = new_count as i64 - self.last_voice_count as i64;
        if change < 0 {
            self.shared_voice_counter
                .fetch_sub((-change) as u64, Ordering::SeqCst);
        } else if change > 0 {
            self.shared_voice_counter
                .fetch_add(change as u64, Ordering::SeqCst);
        }
        self.last_voice_count = new_count;
    }

    pub fn has_voices(&self) -> bool {
        self.voices.has_voices()
    }

    pub fn set_damper(&mut self, damper: bool) {
        self.voices.set_damper(damper);
    }
}
