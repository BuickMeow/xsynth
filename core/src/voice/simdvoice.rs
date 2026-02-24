use std::marker::PhantomData;

use simdeez::prelude::*;

use crate::voice::{ReleaseType, VoiceControlData};

use super::{
    SIMDSample, SIMDSampleMono, SIMDSampleStereo, SIMDVoiceGenerator, VoiceGeneratorBase,
    VoiceSampleGenerator,
};

pub struct SIMDStereoVoice<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>> {
    generator: T,
    remainder: SIMDSampleStereo<S>,
    remainder_pos: usize,
    _s: PhantomData<S>,
}

impl<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>> SIMDStereoVoice<S, T> {
    pub fn new(generator: T) -> SIMDStereoVoice<S, T> {
        SIMDStereoVoice {
            generator,
            remainder: SIMDSampleStereo::<S>::zero(),
            remainder_pos: S::Vf32::WIDTH,
            _s: PhantomData,
        }
    }
}

impl<S, T> VoiceGeneratorBase for SIMDStereoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.generator.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.generator.signal_release(rel_type)
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.generator.process_controls(control)
    }
}

impl<S, T> VoiceSampleGenerator for SIMDStereoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn render_to(&mut self, buffer: &mut [f32]) {
        simd_invoke!(S, {
            let width = S::Vf32::WIDTH;
            let mut buf_idx = 0;
            let buf_len = buffer.len();
            
            // First, consume any remainder from previous call
            while buf_idx < buf_len && self.remainder_pos < width {
                unsafe {
                    *buffer.get_unchecked_mut(buf_idx) += self.remainder.0.get_unchecked(self.remainder_pos);
                    *buffer.get_unchecked_mut(buf_idx + 1) += self.remainder.1.get_unchecked(self.remainder_pos);
                }
                buf_idx += 2;
                self.remainder_pos += 1;
            }
            
            // Stereo has interleaved L/R, so we need to process samples individually
            // But we can still benefit from batching generator calls
            let samples_per_batch = width * 2;
            while buf_idx + samples_per_batch <= buf_len {
                let sample = self.generator.next_sample();
                unsafe {
                    let buf_ptr = buffer.as_mut_ptr().add(buf_idx);
                    for i in 0..width {
                        *buf_ptr.add(i * 2) += sample.0.get_unchecked(i);
                        *buf_ptr.add(i * 2 + 1) += sample.1.get_unchecked(i);
                    }
                }
                buf_idx += samples_per_batch;
            }
            
            // Handle remaining samples
            if buf_idx < buf_len {
                self.remainder = self.generator.next_sample();
                self.remainder_pos = 0;
                while buf_idx < buf_len {
                    unsafe {
                        *buffer.get_unchecked_mut(buf_idx) += self.remainder.0.get_unchecked(self.remainder_pos);
                        *buffer.get_unchecked_mut(buf_idx + 1) += self.remainder.1.get_unchecked(self.remainder_pos);
                    }
                    buf_idx += 2;
                    self.remainder_pos += 1;
                }
            }
        })
    }
}

pub struct SIMDMonoVoice<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>> {
    generator: T,
    remainder: SIMDSampleMono<S>,
    remainder_pos: usize,
    _s: PhantomData<S>,
}

impl<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>> SIMDMonoVoice<S, T> {
    pub fn new(generator: T) -> SIMDMonoVoice<S, T> {
        SIMDMonoVoice {
            generator,
            remainder: SIMDSampleMono::<S>::zero(),
            remainder_pos: S::Vf32::WIDTH,
            _s: PhantomData,
        }
    }
}

impl<S, T> VoiceGeneratorBase for SIMDMonoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.generator.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.generator.signal_release(rel_type)
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.generator.process_controls(control)
    }
}

impl<S, T> VoiceSampleGenerator for SIMDMonoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn render_to(&mut self, buffer: &mut [f32]) {
        simd_invoke!(S, {
            let width = S::Vf32::WIDTH;
            let mut buf_idx = 0;
            let buf_len = buffer.len();
            
            // First, consume any remainder from previous call
            while buf_idx < buf_len && self.remainder_pos < width {
                unsafe {
                    *buffer.get_unchecked_mut(buf_idx) += self.remainder.0.get_unchecked(self.remainder_pos);
                }
                buf_idx += 1;
                self.remainder_pos += 1;
            }
            
            // Process SIMD batches using SIMD load/add/store
            while buf_idx + width <= buf_len {
                let sample = self.generator.next_sample();
                unsafe {
                    let buf_ptr = buffer.as_mut_ptr().add(buf_idx);
                    let dst = S::Vf32::load_from_ptr_unaligned(buf_ptr);
                    (dst + sample.0).copy_to_ptr_unaligned(buf_ptr);
                }
                buf_idx += width;
            }
            
            // Handle remaining samples
            if buf_idx < buf_len {
                self.remainder = self.generator.next_sample();
                self.remainder_pos = 0;
                while buf_idx < buf_len {
                    unsafe {
                        *buffer.get_unchecked_mut(buf_idx) += self.remainder.0.get_unchecked(self.remainder_pos);
                    }
                    buf_idx += 1;
                    self.remainder_pos += 1;
                }
            }
        })
    }
}
