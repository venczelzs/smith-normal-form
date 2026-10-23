//! Smith normal form of an integer matrix.
//!
//! For any `m x n` matrix `A` over the integers there are unimodular matrices
//! `U` (`m x m`) and `V` (`n x n`) such that `U * A * V = D`, where `D` is
//! diagonal with non-negative entries `d_1 | d_2 | ... | d_r` followed by
//! zeros. The `d_i` are the invariant factors and are unique.

use std::cmp::min;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

pub type Matrix = Vec<Vec<BigInt>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmithNormalForm {
    /// The invariant factors `d_1 | d_2 | ... | d_r`, all strictly positive.
    /// Their count is the rank of the matrix.
    pub invariant_factors: Vec<BigInt>,
    /// Unimodular left transform, `rows x rows`.
    pub left: Matrix,
    /// Unimodular right transform, `cols x cols`.
    pub right: Matrix,
    pub rows: usize,
    pub cols: usize,
}

impl SmithNormalForm {
    pub fn rank(&self) -> usize {
        self.invariant_factors.len()
    }

    /// The diagonal form `D` itself, as a full `rows x cols` matrix.
    pub fn diagonal(&self) -> Matrix {
        let mut d = vec![vec![BigInt::zero(); self.cols]; self.rows];
        for (i, factor) in self.invariant_factors.iter().enumerate() {
            d[i][i] = factor.clone();
        }
        d
    }
}

/// Computes the Smith normal form of `a` together with the transforms that
/// realise it: `left * a * right == diagonal()`.
///
/// Panics if the rows of `a` do not all have the same length.
///
/// The algorithm is the classic row/column Euclidean reduction. It always
/// pivots on a smallest-magnitude entry, which keeps intermediate growth
/// modest, and uses arbitrary-precision integers (`BigInt`) to prevent overflow.
pub fn smith_normal_form(a: &[Vec<BigInt>]) -> SmithNormalForm {
    let rows = a.len();
    let cols = a.first().map_or(0, Vec::len);
    assert!(
        a.iter().all(|row| row.len() == cols),
        "matrix rows must all have the same length"
    );

    let mut d: Matrix = a.to_vec();
    let mut left = identity(rows);
    let mut right = identity(cols);

    for t in 0..min(rows, cols) {
        if !move_smallest_nonzero_to_pivot(&mut d, &mut left, &mut right, t) {
            break; // the remaining submatrix is entirely zero
        }

        loop {
            clear_column_below_pivot(&mut d, &mut left, t);
            clear_row_right_of_pivot(&mut d, &mut right, t);

            // A column swap while clearing the row can refill the column, and
            // vice versa, so keep going until both are clean.
            let cleared = (t + 1..rows).all(|i| d[i][t].is_zero())
                && (t + 1..cols).all(|j| d[t][j].is_zero());
            if !cleared {
                continue;
            }

            // The pivot must divide the whole remaining submatrix, otherwise
            // the divisibility chain would fail. Folding an offending row into
            // the pivot row lets the next pass shrink the pivot to a gcd.
            match find_entry_not_divisible_by_pivot(&d, t) {
                Some(i) => add_row_multiple(&mut d, &mut left, t, i, &BigInt::one()),
                None => break,
            }
        }

        if d[t][t].is_negative() {
            negate_row(&mut d, &mut left, t);
        }
    }

    let invariant_factors = (0..min(rows, cols))
        .map(|i| d[i][i].clone())
        .take_while(|factor| !factor.is_zero())
        .collect();

    SmithNormalForm { invariant_factors, left, right, rows, cols }
}

/// Ordinary matrix product, handy for checking `left * a * right`.
/// Implemented using straightforward triple-nested loops familiar in C/C++.
pub fn multiply(a: &[Vec<BigInt>], b: &[Vec<BigInt>]) -> Matrix {
    let rows_a = a.len();
    let cols_a = if rows_a > 0 { a[0].len() } else { 0 };
    let rows_b = b.len();
    let cols_b = if rows_b > 0 { b[0].len() } else { 0 };

    assert!(
        cols_a == rows_b,
        "inner dimensions must agree"
    );
    assert!(
        a.iter().all(|row| row.len() == cols_a),
        "all rows in matrix A must have equal length"
    );
    assert!(
        b.iter().all(|row| row.len() == cols_b),
        "all rows in matrix B must have equal length"
    );

    let mut result = vec![vec![BigInt::zero(); cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = BigInt::zero();
            for k in 0..rows_b {
                sum += &a[i][k] * &b[k][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

/// Brings a smallest-magnitude nonzero entry of the submatrix starting at
/// `(t, t)` into the pivot position. Returns false if that submatrix is zero.
fn move_smallest_nonzero_to_pivot(
    d: &mut Matrix,
    left: &mut Matrix,
    right: &mut Matrix,
    t: usize,
) -> bool {
    let rows = d.len();
    let cols = d[0].len();

    let mut best: Option<(usize, usize)> = None;
    for i in t..rows {
        for j in t..cols {
            if !d[i][j].is_zero()
                && best.as_ref().is_none_or(|&(bi, bj)| d[i][j].abs() < d[bi][bj].abs())
            {
                best = Some((i, j));
            }
        }
    }

    let Some((i, j)) = best else { return false };
    if i != t {
        swap_rows(d, left, t, i);
    }
    if j != t {
        swap_cols(d, right, t, j);
    }
    true
}

/// Zeroes column `t` below the pivot by repeated division with remainder.
fn clear_column_below_pivot(d: &mut Matrix, left: &mut Matrix, t: usize) {
    for i in t + 1..d.len() {
        while !d[i][t].is_zero() {
            let quotient = &d[i][t] / &d[t][t];
            add_row_multiple(d, left, i, t, &-quotient);
            if !d[i][t].is_zero() {
                // The remainder is smaller than the pivot: promote it and
                // continue the Euclidean algorithm.
                swap_rows(d, left, t, i);
            }
        }
    }
}

/// Zeroes row `t` to the right of the pivot, mirroring `clear_column_below_pivot`.
fn clear_row_right_of_pivot(d: &mut Matrix, right: &mut Matrix, t: usize) {
    for j in t + 1..d[0].len() {
        while !d[t][j].is_zero() {
            let quotient = &d[t][j] / &d[t][t];
            add_col_multiple(d, right, j, t, &-quotient);
            if !d[t][j].is_zero() {
                swap_cols(d, right, t, j);
            }
        }
    }
}

/// Finds a row of the remaining submatrix holding an entry the pivot does not
/// divide.
fn find_entry_not_divisible_by_pivot(d: &Matrix, t: usize) -> Option<usize> {
    let pivot = &d[t][t];
    (t + 1..d.len()).find(|&i| (t + 1..d[0].len()).any(|j| !(&d[i][j] % pivot).is_zero()))
}

fn identity(n: usize) -> Matrix {
    let mut mat = vec![vec![BigInt::zero(); n]; n];
    for i in 0..n {
        mat[i][i] = BigInt::one();
    }
    mat
}

fn swap_rows(d: &mut Matrix, left: &mut Matrix, i: usize, j: usize) {
    d.swap(i, j);
    left.swap(i, j);
}

fn swap_cols(d: &mut Matrix, right: &mut Matrix, i: usize, j: usize) {
    for row in d.iter_mut() {
        row.swap(i, j);
    }
    for row in right.iter_mut() {
        row.swap(i, j);
    }
}

/// `row[target] += factor * row[source]`, applied to `d` and to `left`.
fn add_row_multiple(
    d: &mut Matrix,
    left: &mut Matrix,
    target: usize,
    source: usize,
    factor: &BigInt,
) {
    for m in [d, left] {
        for k in 0..m[source].len() {
            let term = factor * &m[source][k];
            m[target][k] += term;
        }
    }
}

/// `col[target] += factor * col[source]`, applied to `d` and to `right`.
fn add_col_multiple(
    d: &mut Matrix,
    right: &mut Matrix,
    target: usize,
    source: usize,
    factor: &BigInt,
) {
    for m in [d, right] {
        for row in m.iter_mut() {
            let term = factor * &row[source];
            row[target] += term;
        }
    }
}

fn negate_row(d: &mut Matrix, left: &mut Matrix, i: usize) {
    for m in [d, left] {
        for value in m[i].iter_mut() {
            *value = -&*value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(a: &[&[i64]]) -> Matrix {
        a.iter()
            .map(|row| row.iter().copied().map(BigInt::from).collect())
            .collect()
    }

    fn factors(f: &[i64]) -> Vec<BigInt> {
        f.iter().copied().map(BigInt::from).collect()
    }

    /// Checks the defining properties: `left * a * right` is the diagonal form,
    /// the transforms are unimodular, and the factors form a divisibility chain.
    fn check(a: &[Vec<BigInt>]) -> SmithNormalForm {
        let snf = smith_normal_form(a);
        assert_eq!(
            multiply(&multiply(&snf.left, a), &snf.right),
            snf.diagonal(),
            "left * a * right != diagonal"
        );
        assert_eq!(determinant(&snf.left).abs(), BigInt::one(), "left is not unimodular");
        assert_eq!(determinant(&snf.right).abs(), BigInt::one(), "right is not unimodular");
        for factor in &snf.invariant_factors {
            assert!(factor.is_positive(), "invariant factors must be positive");
        }
        for pair in snf.invariant_factors.windows(2) {
            assert_eq!(&pair[1] % &pair[0], BigInt::zero(), "divisibility chain broken: {pair:?}");
        }
        snf
    }

    /// Cofactor expansion; only used on the small matrices in these tests.
    fn determinant(m: &Matrix) -> BigInt {
        match m.len() {
            0 => BigInt::one(),
            1 => m[0][0].clone(),
            n => (0..n)
                .map(|j| {
                    let minor: Matrix = m[1..]
                        .iter()
                        .map(|row| {
                            row.iter()
                                .enumerate()
                                .filter(|(k, _)| *k != j)
                                .map(|(_, value)| value.clone())
                                .collect()
                        })
                        .collect();
                    let sign = if j % 2 == 0 { BigInt::one() } else { -BigInt::one() };
                    sign * &m[0][j] * determinant(&minor)
                })
                .fold(BigInt::zero(), |acc, x| acc + x),
        }
    }

    #[test]
    fn classic_example() {
        let a = mat(&[
            &[2, 4, 4],
            &[-6, 6, 12],
            &[10, -4, -16],
        ]);
        assert_eq!(check(&a).invariant_factors, factors(&[2, 6, 12]));
    }

    #[test]
    fn already_diagonal_but_not_a_chain() {
        let a = mat(&[&[2, 0], &[0, 3]]);
        assert_eq!(check(&a).invariant_factors, factors(&[1, 6]));
    }

    #[test]
    fn identity_is_its_own_normal_form() {
        let a = identity(4);
        let snf = check(&a);
        assert_eq!(snf.invariant_factors, factors(&[1, 1, 1, 1]));
        assert_eq!(snf.diagonal(), a);
    }

    #[test]
    fn zero_matrix_has_no_invariant_factors() {
        let snf = check(&vec![vec![BigInt::zero(); 3]; 2]);
        assert!(snf.invariant_factors.is_empty());
        assert_eq!(snf.rank(), 0);
    }

    #[test]
    fn rank_deficient_wide_matrix() {
        // Every row is a multiple of the first, so the rank is 1.
        let a = mat(&[
            &[1, 2, 3, 4],
            &[2, 4, 6, 8],
            &[3, 6, 9, 12],
        ]);
        let snf = check(&a);
        assert_eq!(snf.rank(), 1);
        assert_eq!(snf.invariant_factors, factors(&[1]));
    }

    #[test]
    fn tall_matrix() {
        let a = mat(&[&[4, 6], &[6, 9], &[0, 2]]);
        let snf = check(&a);
        assert_eq!(snf.rows, 3);
        assert_eq!(snf.cols, 2);
        // gcd of the entries is 1; gcd of the 2x2 minors (0, 8, 12) is 4.
        assert_eq!(snf.invariant_factors, factors(&[1, 4]));
    }

    #[test]
    fn negative_entries_yield_positive_factors() {
        let a = mat(&[&[-1, 0], &[0, -1]]);
        assert_eq!(check(&a).invariant_factors, factors(&[1, 1]));
    }

    #[test]
    fn singular_matrix_with_repeated_factor() {
        let a = mat(&[
            &[9, 0, 0],
            &[0, 4, 0],
            &[0, 0, 0],
        ]);
        assert_eq!(check(&a).invariant_factors, factors(&[1, 36]));
    }

    #[test]
    fn large_numbers_beyond_i64() {
        use std::str::FromStr;

        let val1 = BigInt::from_str("1000000000000000000000000000000").unwrap();
        let val2 = BigInt::from_str("2000000000000000000000000000000").unwrap();
        let val3 = BigInt::from_str("3000000000000000000000000000000").unwrap();
        let val4 = BigInt::from_str("4000000000000000000000000000000").unwrap();

        let a = vec![
            vec![val1.clone(), val2.clone()],
            vec![val3.clone(), val4.clone()],
        ];
        let snf = check(&a);
        assert_eq!(snf.rank(), 2);
        assert_eq!(snf.invariant_factors[0], val1);
        assert_eq!(snf.invariant_factors[1], val2);
    }

    #[test]
    fn i64_boundary_and_arithmetic_overflow() {
        let max_i64 = BigInt::from(i64::MAX);
        let min_i64 = BigInt::from(i64::MIN);

        // Matrix entries and operations (like subtraction and multiplication)
        // that would immediately overflow standard signed 64-bit integer limits.
        let a = vec![
            vec![max_i64.clone(), &max_i64 * 2],
            vec![min_i64.clone(), &max_i64 + 1],
        ];
        let snf = check(&a);
        assert_eq!(snf.rank(), 2);
    }

    #[test]
    fn extreme_magnitude_values_exceeding_i128() {
        use std::str::FromStr;

        // 50-digit integers far beyond 128-bit limits (~3.4e38)
        let a1 = BigInt::from_str("12345678901234567890123456789012345678901234567890").unwrap();
        let a2 = BigInt::from_str("98765432109876543210987654321098765432109876543210").unwrap();
        let a3 = BigInt::from_str("11111111111111111111111111111111111111111111111111").unwrap();
        let a4 = BigInt::from_str("55555555555555555555555555555555555555555555555555").unwrap();

        let a = vec![
            vec![a1, a2],
            vec![a3, a4],
        ];
        let snf = check(&a);
        assert_eq!(snf.rank(), 2);
    }

    #[test]
    fn large_coprime_values_requiring_multi_step_euclidean_reduction() {
        use std::str::FromStr;

        // Large coprime values forcing multiple division-with-remainder reduction steps
        // with large intermediate products and remainders.
        let p = BigInt::from_str("10000000000000000000000000000000000000000000000019").unwrap();
        let q = BigInt::from_str("10000000000000000000000000000000000000000000000003").unwrap();

        // 1x2 matrix: computing gcd(p, q) = 1 via Euclidean reduction
        let a = vec![vec![p.clone(), q.clone()]];
        let snf = check(&a);
        assert_eq!(snf.rank(), 1);
        assert_eq!(snf.invariant_factors, vec![BigInt::one()]);

        // 2x2 upper triangular matrix with determinant p
        let b = vec![
            vec![p.clone(), q],
            vec![BigInt::zero(), BigInt::one()],
        ];
        let snf_b = check(&b);
        assert_eq!(snf_b.rank(), 2);
        assert_eq!(snf_b.invariant_factors, vec![BigInt::one(), p]);
    }

    #[test]
    fn intermediate_growth_in_3x3_matrix() {
        use std::str::FromStr;

        let scale = BigInt::from_str("100000000000000000000").unwrap(); // 10^20
        let a = vec![
            vec![&scale * 1, &scale * 2, &scale * 3],
            vec![&scale * 4, &scale * 5, &scale * 6],
            vec![&scale * 7, &scale * 8, &scale * 10],
        ];
        let snf = check(&a);
        assert_eq!(snf.rank(), 3);
        assert_eq!(snf.invariant_factors[0], &scale * 1);
        assert_eq!(snf.invariant_factors[1], &scale * 1);
        assert_eq!(snf.invariant_factors[2], &scale * 3);
    }

    #[test]
    fn empty_matrix() {
        let snf = check(&[]);
        assert_eq!(snf.rows, 0);
        assert_eq!(snf.cols, 0);
        assert!(snf.invariant_factors.is_empty());
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn ragged_input_panics() {
        smith_normal_form(&[vec![BigInt::from(1), BigInt::from(2)], vec![BigInt::from(3)]]);
    }
}
