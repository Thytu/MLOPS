"""
Simple demonstrating file for MLFOW
"""

import mlflow
import torch.optim as optim
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

import network
from data_handler import get_dataset

mlflow.set_tracking_uri("sqlite:///mlruns.db")

client = MlflowClient()
create_experiment_id = client.create_experiment("Learning rate impact 3")


client.create_registered_model("MyModelName", {"tagName": "modelTagValue"}, "This is a description")

EPOCHS = 3
BATCH_SIZE = 32


train_loader, test_loader = get_dataset(BATCH_SIZE)

TRAINING_SIZE = len(iter(train_loader)) * BATCH_SIZE
TESTING_SIZE = len(iter(test_loader)) * BATCH_SIZE

for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
    run = client.create_run(create_experiment_id, tags={
        "Activity": "Benchemark",
        "release.user": "Thytu",
        "release.version": "1.0.0"
    })

    client.log_param(run.info.run_id, "learning_rate", lr)
    client.log_param(run.info.run_id, "EPOCHS", EPOCHS)
    client.log_param(run.info.run_id, "batch_size", BATCH_SIZE)

    neuralNet = network.Network(len(next(iter(train_loader))[0][0][0][0])**2, 10)
    optimizer = optim.Adam(neuralNet.parameters(), lr=lr)

    training_accuracies = []
    training_losses = []

    testing_accuracies = []
    testing_losses = []

    for e in range(EPOCHS):
        print(f"Epoch {e}", end="\t")
        nb_accurate, loss = network.train(neuralNet, train_loader, optimizer)
        val_nb_accurate, validation_loss = network.test(neuralNet, test_loader)

        client.log_metric(run.info.run_id, "training_acc", nb_accurate / TRAINING_SIZE * 100)
        client.log_metric(run.info.run_id, "training_loss", loss / TRAINING_SIZE * 100)
        client.log_metric(run.info.run_id, "validation_acc", val_nb_accurate / TESTING_SIZE * 100)
        client.log_metric(run.info.run_id, "validation_loss", validation_loss / TESTING_SIZE * 100)

        training_accuracies.append(nb_accurate / TRAINING_SIZE * 100)
        training_losses.append(loss / TRAINING_SIZE * 100)

        testing_accuracies.append(val_nb_accurate / TESTING_SIZE * 100)
        testing_losses.append(validation_loss / TESTING_SIZE * 100)

        print(f"Accuracy: {nb_accurate / TRAINING_SIZE * 100}\tLoss:{loss / TRAINING_SIZE * 100}\tValidation Acc: {val_nb_accurate / TESTING_SIZE * 100}\tValidation Loss: {validation_loss / TESTING_SIZE * 100}")



    fig = plt.figure()
    plt.plot(list(range(1, len(training_accuracies)+1)), training_accuracies, color='blue')
    plt.plot(list(range(1, len(testing_accuracies)+1)), testing_accuracies, color='red')

    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig("AccuracyEvolv.png")
    client.log_artifact(run.info.run_id, "AccuracyEvolv.png")

    fig = plt.figure()
    plt.plot(list(range(1, len(training_losses)+1)), training_losses, color='blue')
    plt.plot(list(range(1, len(testing_losses)+1)), testing_losses, color='red')

    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig("LossEvolv.png")
    client.log_artifact(run.info.run_id, "LossEvolv.png")

    client.set_terminated(run.info.run_id)
