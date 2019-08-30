from scripts.experiment import Experiment


def main():
    experiment = Experiment()
    y_pred = experiment.run()
    return y_pred


if __name__ == "__main__":
    main()
