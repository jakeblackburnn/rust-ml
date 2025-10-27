use perceptron::Perceptron;
use std::process;

fn main() {
    let mut model = Perceptron::new(3); 

    println!("training model...");
    model.fit("train.dat").unwrap_or_else(|err| {
        eprintln!("Error during training: {}", err);
        process::exit(1);
    });


    println!("evaluating model on training data...");
    model.evaluate("train.dat").unwrap_or_else(|err| {
        eprintln!("Error during evaluation: {}", err);
        process::exit(1);
    });

    println!("evaluating model on testing data...");
    model.evaluate("test.dat").unwrap_or_else(|err| {
        eprintln!("Error during evaluation: {}", err);
        process::exit(1);
    });
}
