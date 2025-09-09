use linear_regression::LinearRegression;
use std::process;

fn main() {

    let mut model = LinearRegression::new(2, 0.0001, 1000000);
    model.print();

    model.fit("train.dat").unwrap_or_else(|err| {
        eprintln!("training failed: {}", err);
        process::exit(1);
    });

    model.print();
}
