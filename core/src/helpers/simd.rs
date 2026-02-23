use simdeez::*; // nuts

use simdeez::prelude::*;

/// Sum the values of `source` to the values of `target`, writing to `target`.
///
/// Uses runtime selected SIMD operations.
/// Panics if source and target have different lengths.
pub fn sum_simd(source: &[f32], target: &mut [f32]) {
    // Ensure both slices have the same length to prevent out-of-bounds access
    let len = source.len().min(target.len());
    if len == 0 {
        return;
    }
    
    // Debug assertion to catch length mismatches in development
    debug_assert_eq!(
        source.len(), 
        target.len(), 
        "sum_simd: source length ({}) != target length ({})", 
        source.len(), 
        target.len()
    );

    simd_runtime_generate!(
        // Altered code from the SIMD example here https://github.com/jackmott/simdeez
        fn sum(source: &[f32], target: &mut [f32]) {
            let len = source.len();
            let mut i = 0;
            
            // Process SIMD-width chunks
            while i + S::Vf32::WIDTH <= len {
                let src = S::Vf32::load_from_slice(&source[i..]);
                let src2 = S::Vf32::load_from_slice(&target[i..]);
                let sum = src + src2;
                sum.copy_to_slice(&mut target[i..]);
                i += S::Vf32::WIDTH;
            }
            
            // Handle remaining elements (less than SIMD width)
            while i < len {
                target[i] += source[i];
                i += 1;
            }
        }
    );

    sum(&source[..len], &mut target[..len]);
}

#[cfg(test)]
mod tests {
    use super::sum_simd;

    #[test]
    fn test_simd_add() {
        let src = vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut dst = vec![0.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        sum_simd(&src, &mut dst);
        assert_eq!(dst, vec![1.0, 3.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }
}
