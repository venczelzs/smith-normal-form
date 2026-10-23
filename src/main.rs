mod smith;

use smith::{Matrix, multiply, smith_normal_form};

fn main() {
    let a: Matrix = vec![
        vec![2, 4, 4],
        vec![-6, 6, 12],
        vec![10, -4, -16],
    ];

    let snf = smith_normal_form(&a);

    print_matrix("A", &a);
    print_matrix("U", &snf.left);
    print_matrix("V", &snf.right);
    print_matrix("D = U * A * V", &snf.diagonal());

    println!("invariant factors: {:?}", snf.invariant_factors);
    println!("rank: {}", snf.rank());
    println!(
        "U * A * V == D: {}",
        multiply(&multiply(&snf.left, &a), &snf.right) == snf.diagonal()
    );
}

fn print_matrix(label: &str, m: &Matrix) {
    println!("{label} =");
    for row in m {
        let cells: Vec<String> = row.iter().map(|value| format!("{value:>5}")).collect();
        println!("  [{}]", cells.join(" "));
    }
}
