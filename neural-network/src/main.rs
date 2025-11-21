use neural_network::network::NeuralNetwork;
use tensor::Tensor;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Load and prepare Iris dataset with train/test split
/// Returns: (x_train, y_train, x_test, y_test)
fn load_iris_data(csv_path: &str) -> Result<(Tensor, Tensor, Tensor, Tensor), Box<dyn std::error::Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;

    // Collect all samples as (features, one_hot_label) pairs
    let mut samples: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    for result in reader.records() {
        let record = result?;

        // Parse 4 features: sepal_length, sepal_width, petal_length, petal_width
        let features: Vec<f32> = (0..4)
            .map(|i| record[i].parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        // Convert string label to one-hot encoding
        // Setosa -> [1, 0, 0], Versicolor -> [0, 1, 0], Virginica -> [0, 0, 1]
        let one_hot = match record[4].trim_matches('"') {
            "Setosa" => vec![1.0, 0.0, 0.0],
            "Versicolor" => vec![0.0, 1.0, 0.0],
            "Virginica" => vec![0.0, 0.0, 1.0],
            label => return Err(format!("Unknown label: {}", label).into()),
        };

        samples.push((features, one_hot));
    }

    // Shuffle samples for random train/test split
    let mut rng = thread_rng();
    samples.shuffle(&mut rng);

    // Split: 80% train (120 samples), 20% test (30 samples)
    let split_idx = (samples.len() as f32 * 0.8) as usize;
    let (train_samples, test_samples) = samples.split_at(split_idx);

    // Flatten train data into tensors
    let mut train_features = Vec::new();
    let mut train_labels = Vec::new();
    for (features, label) in train_samples {
        train_features.extend(features);
        train_labels.extend(label);
    }

    // Flatten test data into tensors
    let mut test_features = Vec::new();
    let mut test_labels = Vec::new();
    for (features, label) in test_samples {
        test_features.extend(features);
        test_labels.extend(label);
    }

    let x_train = Tensor::new(train_features, vec![train_samples.len(), 4]);
    let y_train = Tensor::new(train_labels, vec![train_samples.len(), 3]);
    let x_test = Tensor::new(test_features, vec![test_samples.len(), 4]);
    let y_test = Tensor::new(test_labels, vec![test_samples.len(), 3]);

    Ok((x_train, y_train, x_test, y_test))
}

fn main() {
    println!("=== Neural Network Test ===\n");

    let mut network = NeuralNetwork::new(&[4, 16, 16, 3], 0.01);
    println!("Created network with architecture: [4, 16, 16, 3]");
    println!("Learning rate: 0.01\n");

    // Load and prepare Iris dataset
    println!("\n=== Loading Iris Dataset ===\n");

    // Load dataset with 80/20 train/test split
    // Note: Feature normalization could be added here for potentially better convergence,
    // but Iris features are already in reasonable ranges (0.1-7.9)
    let (x_train, y_train, x_test, y_test) = load_iris_data("iris.csv")
        .expect("Failed to load Iris dataset");

    println!("Training data shape: {:?}", x_train.shape);
    println!("Training labels shape: {:?}", y_train.shape);
    println!("Test data shape: {:?}", x_test.shape);
    println!("Test labels shape: {:?}", y_test.shape);

    println!("\n=== Training ===\n");

    let loss_history = network.train(
        &x_train,
        &y_train,
        10000,
        Some(&x_test),
        Some(&y_test),
        100,
    )
    .expect("Training failed");

    println!("\n=== Training Complete ===");

    // Save loss history to CSV
    NeuralNetwork::save_loss_history(&loss_history, "loss_history.csv")
        .expect("Failed to save loss history");
    println!("Loss history saved to loss_history.csv");

    // Verify loss is decreasing
    let initial_loss = loss_history[0].train_loss;
    let final_loss = loss_history[9999].train_loss;

    if final_loss < initial_loss {
        println!("\n✓ Training loss decreased from {:.6} to {:.6}", initial_loss, final_loss);
    } else {
        println!("\n✗ Warning: Training loss did not decrease!");
    }

    // Show final validation loss if available
    if let Some(final_val_loss) = loss_history.last().and_then(|r| r.val_loss) {
        println!("Final validation loss: {:.6}", final_val_loss);
    }

    println!("\n=== Test Set Evaluation ===");

    // Make predictions on test set
    let predictions = network.predict(&x_test).expect("Prediction failed");

    // Get predicted class indices (argmax along axis 1)
    let predicted_classes = predictions.view().argmax_axis(1).expect("Failed to get predicted classes");

    // Get true class indices from one-hot encoded labels
    let true_classes = y_test.view().argmax_axis(1).expect("Failed to get true classes");

    // Calculate accuracy
    let correct = predicted_classes.iter()
        .zip(true_classes.iter())
        .filter(|(pred, true_class)| pred == true_class)
        .count();
    let total = predicted_classes.len();
    let accuracy = (correct as f64 / total as f64) * 100.0;
    let error_rate = 100.0 - accuracy;

    println!("Test accuracy: {:.2}%", accuracy);
    println!("Test error rate: {:.2}%", error_rate);
}
