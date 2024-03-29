import optuna


def main():
    study_name = "lgb_12"
    study = optuna.study.load_study(
        study_name, f"sqlite:///../database/{study_name}.db"
    )
    df = study.trials_dataframe()[["params", "value"]].sort_values("value")
    print(df.shape[0], "trials")
    print(df.head(10))


if __name__ == "__main__":
    main()
