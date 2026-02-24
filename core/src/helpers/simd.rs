use simdeez::*;

use simdeez::prelude::*;

/// Sum the values of `source` to the values of `target`, writing to `target`.
///
/// Uses runtime selected SIMD operations with aggressive optimization.
/// Panics if source and target have different lengths.
#[inline(always)]
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
        // Highly optimized SIMD sum with loop unrolling
        fn sum(source: &[f32], target: &mut [f32]) {
            let len = source.len();
            let width = S::Vf32::WIDTH;
            let width2 = width * 2;
            let width4 = width * 4;
            let mut i = 0;
            
            // Process 4x SIMD-width chunks for maximum throughput
            while i + width4 <= len {
                unsafe {
                    let src0 = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i));
                    let src1 = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i + width));
                    let src2 = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i + width2));
                    let src3 = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i + width2 + width));
                    
                    let dst0 = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i));
                    let dst1 = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i + width));
                    let dst2 = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i + width2));
                    let dst3 = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i + width2 + width));
                    
                    let sum0 = src0 + dst0;
                    let sum1 = src1 + dst1;
                    let sum2 = src2 + dst2;
                    let sum3 = src3 + dst3;
                    
                    sum0.copy_to_ptr_unaligned(target.as_mut_ptr().add(i));
                    sum1.copy_to_ptr_unaligned(target.as_mut_ptr().add(i + width));
                    sum2.copy_to_ptr_unaligned(target.as_mut_ptr().add(i + width2));
                    sum3.copy_to_ptr_unaligned(target.as_mut_ptr().add(i + width2 + width));
                }
                i += width4;
            }
            
            // Process 2x SIMD-width chunks
            while i + width2 <= len {
                unsafe {
                    let src0 = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i));
                    let src1 = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i + width));
                    let dst0 = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i));
                    let dst1 = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i + width));
                    (src0 + dst0).copy_to_ptr_unaligned(target.as_mut_ptr().add(i));
                    (src1 + dst1).copy_to_ptr_unaligned(target.as_mut_ptr().add(i + width));
                }
                i += width2;
            }
            
            // Process SIMD-width chunks
            while i + width <= len {
                unsafe {
                    let src = S::Vf32::load_from_ptr_unaligned(source.as_ptr().add(i));
                    let dst = S::Vf32::load_from_ptr_unaligned(target.as_ptr().add(i));
                    (src + dst).copy_to_ptr_unaligned(target.as_mut_ptr().add(i));
                }
                i += width;
            }
            
            // Handle remaining elements
            while i < len {
                unsafe {
                    *target.get_unchecked_mut(i) += *source.get_unchecked(i);
                }
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
