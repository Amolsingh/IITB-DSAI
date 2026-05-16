import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def print_diagnostics(df, title="EDA Report", target_col="price"):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # =====================================================
    # DATASET SHAPE / MISSING VALUES (Logic remains same)
    # =====================================================
    rows, cols = df.shape
    print(f"\nDataset Shape: {rows} rows × {cols} columns")
    
    duplicate_count = df.duplicated().sum()
    print(f"\nTotal Duplicate Rows: {duplicate_count}")

    total_missing = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {total_missing}")

    missing_per_column = df.isnull().sum()
    print("\nMissing Values Per Column:")
    missing_found = False
    for col, val in missing_per_column.items():
        if val > 0:
            print(f"{col}: {val}")
            missing_found = True
    if not missing_found:
        print("No Missing Values")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # =====================================================
    # HISTOGRAMS (Standardized grid)
    # =====================================================
    print("\nGenerating Histograms...")
    df[numeric_cols].hist(figsize=(15, 10), bins=30)
    plt.suptitle(f"{title} - Histograms", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # =====================================================
    # BOXPLOTS FOR NUMERIC COLUMNS (Subplot Grid)
    # =====================================================
    print("\nGenerating Boxplots...")
    n_num = len(numeric_cols)
    cols_grid = 3
    rows_grid = (n_num + cols_grid - 1) // cols_grid
    
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(15, rows_grid * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot - {col}")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

    # =====================================================
    # CORRELATION & CONSOLIDATED BOXPLOT (Side-by-Side)
    # =====================================================
    print("\nGenerating Heatmap and Consolidated Boxplot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    ax1.set_title(f"{title} - Correlation Heatmap")

    # Consolidated Boxplot
    consolidated_cols = [c for c in numeric_cols if c != target_col]
    sns.boxplot(data=df[consolidated_cols], ax=ax2)
    ax2.set_title(f"{title} - Consolidated Boxplot")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # COUNT PLOTS FOR CATEGORICAL COLUMNS (Subplot Grid)
    # =====================================================
    if categorical_cols:
        print("\nGenerating Count Plots...")
        n_cat = len(categorical_cols)
        rows_cat = (n_cat + 1) // 2
        fig, axes = plt.subplots(rows_cat, 2, figsize=(15, rows_cat * 5))
        axes = axes.flatten()

        for i, col in enumerate(categorical_cols):
            sns.countplot(x=df[col], order=df[col].value_counts().index, ax=axes[i])
            axes[i].set_title(f"Count Plot - {col}")
            axes[i].tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    print("\nDiagnostics Completed Successfully!")



def split_train_test(df, target_col, test_size=0.2, random_state=42):
    """
    Splits dataframe into train and test sets.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input dataframe

    target_col : str
        Target column name

    test_size : float
        Percentage of test data

    random_state : int
        Random seed for reproducibility

    Returns:
    -------
    X_train, X_test, y_train, y_test
    """

    # ==========================================
    # SEPARATE FEATURES AND TARGET
    # ==========================================

    X = df.drop(columns=[target_col])

    y = df[target_col]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ==========================================
    # PRINT SHAPES
    # ==========================================

    print("\nTrain-Test Split Completed")

    print("\nX_train Shape :", X_train.shape)
    print("X_test Shape  :", X_test.shape)

    print("\ny_train Shape :", y_train.shape)
    print("y_test Shape  :", y_test.shape)

    # ==========================================
    # RETURN
    # ==========================================

    return X_train, X_test, y_train, y_test


def apply_standardization(
    train_df,
    test_df,
    numeric_cols,
    target_col='price'
):
    """
    Standardizes numeric feature columns using StandardScaler.

    This helper function fits the scaler only on the training
    dataset and applies the learned scaling parameters to both
    the training and test datasets.

    The target column is excluded from scaling to prevent
    transformation of the dependent variable.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataframe containing numeric features.

    test_df : pandas.DataFrame
        Test dataframe containing numeric features.

    numeric_cols : list
        List of numeric column names.

    target_col : str, optional
        Target column name to exclude from scaling.
        Default is 'price'.

    Returns
    -------
    train_df : pandas.DataFrame
        Standardized training dataframe.

    test_df : pandas.DataFrame
        Standardized test dataframe.
    """

    scaler = StandardScaler()

    cols_to_scale = [
        c for c in numeric_cols
        if c != target_col
    ]

    train_df[cols_to_scale] = scaler.fit_transform(
        train_df[cols_to_scale]
    )

    test_df[cols_to_scale] = scaler.transform(
        test_df[cols_to_scale]
    )

    return train_df, test_df


def feature_engineer_numeric_only(df):
    """
    Performs numeric-only feature engineering for the prediction pipeline.

    Workflow:
    ----------
    1. Copy input dataframe
    2. Replace invalid zero values with NaN
    3. Create volume feature using x * y * z
    4. Drop categorical columns:
    5. Return only numeric feature columns required for modeling

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features.

    Returns
    -------
    pandas.DataFrame: Dataframe containing only numeric features:
    """

    # ==========================================
    # COPY DATAFRAME
    # ==========================================

    df_processed = df.copy()

    # ==========================================
    # REPLACE INVALID ZERO VALUES
    # ==========================================

    df_processed[['x', 'y', 'z']] = df_processed[
        ['x', 'y', 'z']
    ].replace(0, np.nan)

    # ==========================================
    # CREATE VOLUME FEATURE
    # ==========================================

    df_processed['volume'] = (
        df_processed['x']
        * df_processed['y']
        * df_processed['z']
    )

    # ==========================================
    # DROP CATEGORICAL COLUMNS
    # ==========================================

    df_processed = df_processed.drop(
        columns=['cut', 'color', 'clarity'],
        errors='ignore'
    )

    # ==========================================
    # RETURN REQUIRED NUMERIC FEATURES
    # ==========================================

    numeric_features = [
        'carat',
        'depth',
        'table',
        'x',
        'y',
        'z',
        'volume'
    ]

    return df_processed[numeric_features]


def preprocess_diamond_log(df):
    """
    Performs manual preprocessing to visualize the effect
    of log transformation and standardization on the
    training data distribution.

    Workflow:
    ----------
    1. Copy original dataframe
    2. Replace invalid zero values in x, y, z with NaN
    3. Create volume feature
    4. Drop categorical columns
    5. Split data into train and test sets
    6. Apply log transformation on independent numeric features
    7. Print diagnostics after log transformation
    8. Apply standardization using train data only
    9. Print diagnostics after scaling

    Parameters
    ----------
    df : pandas.DataFrame
        Original cleaned diamond dataframe.

    Returns
    -------
    train_df : pandas.DataFrame
        Processed training dataframe.

    test_df : pandas.DataFrame
        Processed test dataframe.
    """

    # ==========================================
    # COPY ORIGINAL DATAFRAME
    # ==========================================

    df_processed = df.copy()

    # ==========================================
    # REPLACE INVALID ZERO VALUES
    # ==========================================

    df_processed[['x', 'y', 'z']] = df_processed[
        ['x', 'y', 'z']
    ].replace(0, np.nan)

    # ==========================================
    # CREATE VOLUME FEATURE
    # ==========================================

    df_processed['volume'] = (
        df_processed['x']
        * df_processed['y']
        * df_processed['z']
    )

    # ==========================================
    # DROP CATEGORICAL COLUMNS
    # ==========================================

    df_processed = df_processed.drop(
        columns=['cut', 'color', 'clarity']
    )

    # ==========================================
    # GET NUMERIC COLUMNS
    # ==========================================

    numeric_cols = df_processed.select_dtypes(
        include=np.number
    ).columns.tolist()

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = split_train_test(
        df_processed,
        target_col='price'
    )

    # ==========================================
    # COMBINE TRAIN AND TEST
    # ==========================================

    train_df = pd.concat(
        [X_train, y_train],
        axis=1
    )

    test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )

    # ==========================================
    # APPLY LOG TRANSFORMATION
    # ==========================================

    cols_to_log = [
        c for c in numeric_cols
        if c != 'price'
    ]

    train_df[cols_to_log] = np.log1p(
        train_df[cols_to_log]
    )

    test_df[cols_to_log] = np.log1p(
        test_df[cols_to_log]
    )

    # ==========================================
    # DIAGNOSTICS AFTER LOG TRANSFORM
    # ==========================================

    print_diagnostics(
        train_df,
        title="Train (Post Log, Pre Scaling)",
        target_col='price'
    )

    # ==========================================
    # APPLY STANDARDIZATION
    # ==========================================

    train_df, test_df = apply_standardization(
        train_df,
        test_df,
        numeric_cols=numeric_cols,
        target_col='price'
    )

    # ==========================================
    # DIAGNOSTICS AFTER SCALING
    # ==========================================

    print_diagnostics(
        train_df,
        title="Train (Post Log & Scaling)",
        target_col='price'
    )

    return train_df, test_df



def calculate_regression_metrics(
    y_true,
    y_pred,
    dataset_name="Dataset"
):
    """
    Calculates and prints regression evaluation metrics.

    Metrics:
    --------
    - RMSE
    - MAE
    - R² Score

    Parameters
    ----------
    y_true : array-like
        Actual target values.

    y_pred : array-like
        Predicted target values.

    dataset_name : str
        Name of dataset being evaluated.
    """

    # ==========================================
    # CALCULATE METRICS
    # ==========================================

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    # ==========================================
    # PRINT METRICS
    # ==========================================

    print("\n" + "=" * 50)
    print(f"{dataset_name} Metrics")
    print("=" * 50)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    dataset_name="Dataset"
):
    """
    Creates Actual vs Predicted plot.
    """

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.title(
        f"{dataset_name} - Actual vs Predicted"
    )

    plt.show()


def plot_residuals_vs_predicted(
    y_true,
    y_pred,
    dataset_name="Dataset"
):
    """
    Creates Residuals vs Predicted plot.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle='--'
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    plt.title(
        f"{dataset_name} - Residuals vs Predicted"
    )

    plt.show()


def plot_qq_residuals(
    y_true,
    y_pred,
    dataset_name="Dataset"
):
    """
    Creates QQ plot for residual analysis.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(7, 6))

    stats.probplot(
        residuals,
        dist="norm",
        plot=plt
    )

    plt.title(
        f"{dataset_name} - QQ Plot of Residuals"
    )

    plt.show()


def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
    """
    Runs complete regression evaluation with consolidated subplots.
    """

    # 1. Calculate and print metrics (logic unchanged)
    calculate_regression_metrics(y_true, y_pred, dataset_name)

    # 2. Create a figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 3. Pass each axis to the plotting functions
    plot_actual_vs_predicted(y_true, y_pred, dataset_name, ax=axes[0])
    plot_residuals_vs_predicted(y_true, y_pred, dataset_name, ax=axes[1])
    plot_qq_residuals(y_true, y_pred, dataset_name, ax=axes[2])

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, dataset_name="Dataset", ax=None):
    """Creates Actual vs Predicted plot on a specific axis."""
    if ax is None:
        plt.figure(figsize=(7, 6))
        ax = plt.gca()

    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{dataset_name} - Actual vs Predicted")

def plot_residuals_vs_predicted(y_true, y_pred, dataset_name="Dataset", ax=None):
    """Creates Residuals vs Predicted plot on a specific axis."""
    residuals = y_true - y_pred
    if ax is None:
        plt.figure(figsize=(7, 6))
        ax = plt.gca()

    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{dataset_name} - Residuals vs Predicted")

def plot_qq_residuals(y_true, y_pred, dataset_name="Dataset", ax=None):
    """Creates QQ plot for residual analysis on a specific axis."""
    residuals = y_true - y_pred
    if ax is None:
        plt.figure(figsize=(7, 6))
        ax = plt.gca()

    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{dataset_name} - QQ Plot of Residuals")

def run_pipeline_log(df, target_col):

    # ==========================================
    # COPY DATAFRAME
    # ==========================================

    df_pipeline = df.copy()

    # ==========================================
    # SEPARATE FEATURES & TARGET
    # ==========================================

    X = df_pipeline.drop(columns=[target_col])

    y = df_pipeline[target_col]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================
    # BUILD PIPELINE
    # ==========================================

    pipeline = SkPipeline([

        (
            'feature_engineering',

            FunctionTransformer(
                feature_engineer_numeric_only
            )
        ),

        (
            'imputer',

            SimpleImputer(strategy='median')
        ),

        (
            'log_transform',

            FunctionTransformer(np.log1p)
        ),

        (
            'scaler',
            
            StandardScaler()
        ),

        (
            'model',

            RandomForestRegressor(
                random_state=42
            )
        )
    ])

    # ==========================================
    # FIT PIPELINE
    # ==========================================

    pipeline.fit(X_train, y_train)

    # ==========================================
    # TRAIN PREDICTIONS
    # ==========================================

    y_train_pred = pipeline.predict(X_train)

    # ==========================================
    # TEST PREDICTIONS
    # ==========================================

    y_test_pred = pipeline.predict(X_test)

    # ==========================================
    # TRAIN EVALUATION
    # ==========================================

    evaluate_model(
        y_train,
        y_train_pred,
        dataset_name="Training Data"
    )

    # ==========================================
    # TEST EVALUATION
    # ==========================================

    evaluate_model(
        y_test,
        y_test_pred,
        dataset_name="Test Data"
    )

    return pipeline