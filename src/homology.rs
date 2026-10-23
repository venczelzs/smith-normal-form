//! Simplicial homology and chain complex homology computations.
//!
//! Homology groups of a chain complex or simplicial complex are computed over
//! the integers using the Smith normal form decomposition from `smith.rs`.
//!
//! For a chain complex with boundary operators $\partial_k: C_k \to C_{k-1}$,
//! the $k$-th homology group $H_k = \ker(\partial_k) / \operatorname{im}(\partial_{k+1})$
//! decomposes into a free part and a torsion part:
//! $$H_k \cong \mathbb{Z}^{\beta_k} \oplus \bigoplus_{i} \mathbb{Z} / d_i \mathbb{Z}$$
//! where $\beta_k = \dim(\ker \partial_k) - \operatorname{rank}(\operatorname{im} \partial_{k+1})$
//! is the $k$-th Betti number, and $d_i > 1$ are the invariant factors of $\partial_{k+1}$.

use std::collections::{BTreeSet, HashMap};
use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::smith::{Matrix, smith_normal_form};

/// Representation of a finitely generated homology group $H_k \cong \mathbb{Z}^\beta \oplus \bigoplus_i \mathbb{Z}/d_i\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomologyGroup {
    /// Dimension / degree $k$.
    pub dimension: usize,
    /// Betti number $\beta_k$, the rank of the free part $\mathbb{Z}^{\beta_k}$.
    pub betti_number: usize,
    /// Torsion coefficients $d_i > 1$ such that $d_1 | d_2 | \dots | d_m$.
    pub torsion_coefficients: Vec<BigInt>,
}

impl HomologyGroup {
    /// Returns true if the homology group is trivial (rank 0 and no torsion).
    pub fn is_trivial(&self) -> bool {
        self.betti_number == 0 && self.torsion_coefficients.is_empty()
    }
}

/// An abstract finite chain complex over $\mathbb{Z}$.
///
/// $C_n \xrightarrow{\partial_n} C_{n-1} \dots \xrightarrow{\partial_1} C_0 \xrightarrow{\partial_0} 0$.
#[derive(Debug, Clone)]
pub struct ChainComplex {
    /// Dimensions of chain groups $C_0, C_1, \dots, C_n$.
    pub dimensions: Vec<usize>,
    /// Boundary matrices $\partial_1, \partial_2, \dots, \partial_n$.
    /// `boundaries[k - 1]` is the matrix $\partial_k: C_k \to C_{k-1}$ of size `dim(C_{k-1}) x dim(C_k)`.
    pub boundaries: Vec<Matrix>,
}

impl ChainComplex {
    /// Creates a chain complex from chain group dimensions and boundary matrices.
    ///
    /// Validates that dimensions match matrix shapes.
    pub fn new(dimensions: Vec<usize>, boundaries: Vec<Matrix>) -> Self {
        assert_eq!(
            dimensions.len(),
            boundaries.len() + 1,
            "dimensions length must be boundaries length + 1"
        );
        for (k, d_k) in boundaries.iter().enumerate() {
            let from_dim = dimensions[k + 1];
            let to_dim = dimensions[k];
            assert_eq!(
                d_k.len(),
                to_dim,
                "boundary matrix {} row count must match target dimension",
                k + 1
            );
            for row in d_k {
                assert_eq!(
                    row.len(),
                    from_dim,
                    "boundary matrix {} column count must match source dimension",
                    k + 1
                );
            }
        }
        Self {
            dimensions,
            boundaries,
        }
    }

    /// Computes the homology groups $H_0, H_1, \dots, H_n$ of this chain complex.
    pub fn compute_homology(&self) -> Vec<HomologyGroup> {
        let n = self.dimensions.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute Smith normal forms for all boundary operators d_1, ..., d_{n-1}.
        // snf[k] corresponds to \partial_{k+1}: C_{k+1} -> C_k
        let snfs: Vec<_> = self.boundaries.iter().map(|mat| smith_normal_form(mat)).collect();

        let mut groups = Vec::with_capacity(n);

        for k in 0..n {
            let dim_c_k = self.dimensions[k];

            // rank of \partial_k: C_k -> C_{k-1}
            let rank_d_k = if k == 0 {
                0
            } else {
                snfs[k - 1].rank()
            };

            // dim(ker \partial_k) = dim(C_k) - rank(\partial_k)
            let nullity_d_k = dim_c_k.checked_sub(rank_d_k).expect("rank cannot exceed dimension");

            // rank of \partial_{k+1}: C_{k+1} -> C_k
            let rank_d_kp1 = if k < self.boundaries.len() {
                snfs[k].rank()
            } else {
                0
            };

            // Betti number = nullity(d_k) - rank(d_{k+1})
            let betti_number = nullity_d_k
                .checked_sub(rank_d_kp1)
                .expect("im(d_{k+1}) must be a subspace of ker(d_k)");

            // Torsion coefficients come from invariant factors > 1 of \partial_{k+1}
            let torsion_coefficients = if k < self.boundaries.len() {
                snfs[k]
                    .invariant_factors
                    .iter()
                    .filter(|&f| !f.is_one())
                    .cloned()
                    .collect()
            } else {
                Vec::new()
            };

            groups.push(HomologyGroup {
                dimension: k,
                betti_number,
                torsion_coefficients,
            });
        }

        groups
    }
}

/// An abstract simplicial complex defined as a set of simplices closed under inclusion.
#[derive(Debug, Clone, Default)]
pub struct SimplicialComplex {
    /// Simplices stored as sorted vertex vectors.
    simplices: BTreeSet<Vec<usize>>,
}

impl SimplicialComplex {
    /// Creates an empty simplicial complex.
    pub fn new() -> Self {
        Self {
            simplices: BTreeSet::new(),
        }
    }

    /// Adds a simplex and all of its non-empty faces to the simplicial complex.
    pub fn add_simplex(&mut self, mut vertices: Vec<usize>) {
        vertices.sort_unstable();
        vertices.dedup();
        if vertices.is_empty() {
            return;
        }
        self.add_faces(&vertices);
    }

    fn add_faces(&mut self, vertices: &[usize]) {
        let n = vertices.len();
        if n == 0 {
            return;
        }
        // Add all 2^n - 1 non-empty subsets
        let total = 1usize << n;
        for mask in 1..total {
            let mut face = Vec::new();
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    face.push(vertices[i]);
                }
            }
            self.simplices.insert(face);
        }
    }

    /// Returns the maximum dimension of simplices in the complex, or None if empty.
    pub fn max_dimension(&self) -> Option<usize> {
        self.simplices.iter().map(|s| s.len().saturating_sub(1)).max()
    }

    /// Returns all $k$-simplices in canonical sorted order.
    pub fn simplices_of_dimension(&self, k: usize) -> Vec<Vec<usize>> {
        self.simplices
            .iter()
            .filter(|s| s.len() == k + 1)
            .cloned()
            .collect()
    }

    /// Builds the chain complex $C_\bullet(K)$ associated with this simplicial complex.
    pub fn to_chain_complex(&self) -> ChainComplex {
        let max_dim = match self.max_dimension() {
            Some(d) => d,
            None => return ChainComplex::new(vec![0], Vec::new()),
        };

        let mut simplices_by_dim: Vec<Vec<Vec<usize>>> = Vec::with_capacity(max_dim + 1);
        for k in 0..=max_dim {
            simplices_by_dim.push(self.simplices_of_dimension(k));
        }

        let dimensions: Vec<usize> = simplices_by_dim.iter().map(|s| s.len()).collect();
        let mut boundaries = Vec::with_capacity(max_dim);

        for k in 1..=max_dim {
            let lower_simplices = &simplices_by_dim[k - 1];
            let upper_simplices = &simplices_by_dim[k];

            let lower_index_map: HashMap<&[usize], usize> = lower_simplices
                .iter()
                .enumerate()
                .map(|(idx, s)| (s.as_slice(), idx))
                .collect();

            let rows = lower_simplices.len();
            let cols = upper_simplices.len();
            let mut matrix = vec![vec![BigInt::zero(); cols]; rows];

            for (col, simplex) in upper_simplices.iter().enumerate() {
                // \partial_k([v_0, ..., v_k]) = \sum_{i=0}^k (-1)^i [v_0, ..., \hat{v}_i, ..., v_k]
                for (i, _) in simplex.iter().enumerate() {
                    let mut face = Vec::with_capacity(k);
                    for (j, &v) in simplex.iter().enumerate() {
                        if j != i {
                            face.push(v);
                        }
                    }
                    if let Some(&row) = lower_index_map.get(face.as_slice()) {
                        let sign = if i % 2 == 0 {
                            BigInt::one()
                        } else {
                            -BigInt::one()
                        };
                        matrix[row][col] += sign;
                    }
                }
            }

            boundaries.push(matrix);
        }

        ChainComplex::new(dimensions, boundaries)
    }

    /// Computes the simplicial homology groups of this complex.
    pub fn compute_homology(&self) -> Vec<HomologyGroup> {
        self.to_chain_complex().compute_homology()
    }
}

/// Convenience function to compute simplicial homology directly from a list of maximal simplices.
pub fn compute_simplicial_homology(facets: &[Vec<usize>]) -> Vec<HomologyGroup> {
    let mut complex = SimplicialComplex::new();
    for facet in facets {
        complex.add_simplex(facet.clone());
    }
    complex.compute_homology()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point() {
        // A single vertex: H_0 = Z, all higher are 0.
        let homology = compute_simplicial_homology(&[vec![0]]);
        assert_eq!(homology.len(), 1);
        assert_eq!(homology[0].betti_number, 1);
        assert!(homology[0].torsion_coefficients.is_empty());
    }

    #[test]
    fn two_disconnected_points() {
        // Two disconnected vertices: H_0 = Z^2.
        let homology = compute_simplicial_homology(&[vec![0], vec![1]]);
        assert_eq!(homology.len(), 1);
        assert_eq!(homology[0].betti_number, 2);
    }

    #[test]
    fn interval() {
        // A single edge: contractible, H_0 = Z, H_1 = 0.
        let homology = compute_simplicial_homology(&[vec![0, 1]]);
        assert_eq!(homology.len(), 2);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 0);
    }

    #[test]
    fn triangle_boundary_circle_s1() {
        // Circle S^1 as boundary of a triangle: edges (0,1), (1,2), (0,2).
        // H_0 = Z, H_1 = Z.
        let homology = compute_simplicial_homology(&[vec![0, 1], vec![1, 2], vec![0, 2]]);
        assert_eq!(homology.len(), 2);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 1);
        assert!(homology[1].torsion_coefficients.is_empty());
    }

    #[test]
    fn filled_triangle_2_simplex() {
        // Solid triangle: H_0 = Z, H_1 = 0, H_2 = 0.
        let homology = compute_simplicial_homology(&[vec![0, 1, 2]]);
        assert_eq!(homology.len(), 3);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 0);
        assert_eq!(homology[2].betti_number, 0);
    }

    #[test]
    fn sphere_s2_tetrahedron_boundary() {
        // Boundary of a tetrahedron: 4 triangular faces.
        // H_0 = Z, H_1 = 0, H_2 = Z.
        let homology = compute_simplicial_homology(&[
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ]);
        assert_eq!(homology.len(), 3);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 0);
        assert_eq!(homology[2].betti_number, 1);
        assert!(homology[2].torsion_coefficients.is_empty());
    }

    #[test]
    fn torus_t2_triangulation() {
        // Standard minimal 14-triangle triangulation of a 2-torus T^2 with 7 vertices
        // (Császár / standard Möbius grid mod 7):
        // H_0 = Z, H_1 = Z^2, H_2 = Z.
        let mut torus_facets = Vec::new();
        for i in 0..7 {
            torus_facets.push(vec![i, (i + 1) % 7, (i + 3) % 7]);
            torus_facets.push(vec![(i + 1) % 7, (i + 4) % 7, (i + 3) % 7]);
        }

        let homology = compute_simplicial_homology(&torus_facets);
        assert_eq!(homology.len(), 3);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 2);
        assert_eq!(homology[2].betti_number, 1);
        assert!(homology[0].torsion_coefficients.is_empty());
        assert!(homology[1].torsion_coefficients.is_empty());
        assert!(homology[2].torsion_coefficients.is_empty());
    }

    #[test]
    fn real_projective_plane_rp2() {
        // Standard minimal 6-vertex 10-triangle triangulation of RP^2:
        // H_0 = Z, H_1 = Z/2Z (betti = 0, torsion = [2]), H_2 = 0.
        let rp2_facets = vec![
            vec![0, 1, 2],
            vec![1, 2, 3],
            vec![1, 3, 4],
            vec![2, 3, 5],
            vec![0, 2, 4],
            vec![0, 3, 4],
            vec![0, 1, 5],
            vec![1, 4, 5],
            vec![0, 3, 5],
            vec![2, 4, 5],
        ];

        let homology = compute_simplicial_homology(&rp2_facets);
        assert_eq!(homology.len(), 3);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 0);
        assert_eq!(homology[1].torsion_coefficients, vec![BigInt::from(2)]);
        assert_eq!(homology[2].betti_number, 0);
        assert!(homology[2].torsion_coefficients.is_empty());
    }

    #[test]
    fn klein_bottle() {
        // Standard 9-vertex 18-triangle triangulation of the Klein bottle.
        // H_0 = Z, H_1 = Z + Z/2Z (betti = 1, torsion = [2]), H_2 = 0.
        // Grid [0..3] x [0..3] with twisted identifications:
        // Vertices 0..8: (x, y) = x + 3*y where x in 0..2, y in 0..2.
        // (x+3, y) ~ (x, y); (x, y+3) ~ (2-x, y)
        let v = |x: usize, y: usize| -> usize {
            let x_mod = x % 3;
            let y_mod = y % 3;
            let final_x = if (y / 3) % 2 == 1 {
                (2 - x_mod) % 3
            } else {
                x_mod
            };
            final_x + 3 * y_mod
        };

        let mut klein_facets = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                let v00 = v(x, y);
                let v10 = v(x + 1, y);
                let v01 = v(x, y + 1);
                let v11 = v(x + 1, y + 1);
                klein_facets.push(vec![v00, v10, v11]);
                klein_facets.push(vec![v00, v11, v01]);
            }
        }

        let homology = compute_simplicial_homology(&klein_facets);
        assert_eq!(homology.len(), 3);
        assert_eq!(homology[0].betti_number, 1);
        assert_eq!(homology[1].betti_number, 1);
        assert_eq!(homology[1].torsion_coefficients, vec![BigInt::from(2)]);
        assert_eq!(homology[2].betti_number, 0);
    }
}
