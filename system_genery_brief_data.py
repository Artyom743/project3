import numpy as np
import scipy.stats as stats
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

def load_boston_data():
    """
    Load Boston housing dataset using fetch_openml since it was removed from sklearn.
    
    Returns
    -------
    tuple
        Features (X) as pandas DataFrame and target (y) as pandas Series
    """
    try:
        # Boston dataset was removed from sklearn, using fetch_openml instead
        boston = fetch_openml(name='boston', version=1, as_frame=True)
        X = boston.frame.drop('MEDV', axis=1)  # MEDV is the target variable
        y = boston.frame['MEDV']
        
        return X, y
        
    except Exception as e:
        print(f"Error loading Boston dataset: {e}")
        return None, None

def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable array
    y : np.ndarray
        Second variable array

    Returns
    -------
    float
        Pearson correlation coefficient

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty
    """
    try:
        if len(x) != len(y):
            raise ValueError("Input arrays must have same length")
        if len(x) == 0:
            raise ValueError("Input arrays cannot be empty")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")

        pearson_corr, _ = stats.pearsonr(x, y)

        return pearson_corr
        
    except ValueError as ve:
        print(f"Value error in Pearson correlation: {ve}")
        return np.nan
    except TypeError as te:
        print(f"Type error in Pearson correlation: {te}")
        return np.nan
    except Exception as e:
        print(f"Unexpected error calculating Pearson correlation: {e}")
        return np.nan
    
def calculate_spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Spearman correlation coefficient between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable array
    y : np.ndarray
        Second variable array

    Returns
    -------
    float
        Spearman correlation coefficient

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty
    """
    try:
        if len(x) != len(y):
            raise ValueError("Input arrays must have same length")
        if len(x) == 0:
            raise ValueError("Input arrays cannot be empty")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")

        spearman_corr, _ = stats.spearmanr(x, y)

        return spearman_corr
        
    except ValueError as ve:
        print(f"Value error in Spearman correlation: {ve}")
        return np.nan
    except TypeError as te:
        print(f"Type error in Spearman correlation: {te}")
        return np.nan
    except Exception as e:
        print(f"Unexpected error calculating Spearman correlation: {e}")
        return np.nan

def analyze_correlation_difference(spearman_corr: float, pearson_corr: float) -> str:
    """
    Analyze the difference between Spearman and Pearson correlation coefficients.

    Parameters
    ----------
    spearman_corr : float
        Spearman correlation coefficient
    pearson_corr : float
        Pearson correlation coefficient

    Returns
    -------
    str
        Explanation of the difference between correlation coefficients

    Raises
    ------
    ValueError
        If correlation coefficients are NaN
    """
    try:
        if np.isnan(spearman_corr) or np.isnan(pearson_corr):
            raise ValueError("Correlation coefficients cannot be NaN")
        if not isinstance(spearman_corr, (int, float)) or not isinstance(pearson_corr, (int, float)):
            raise TypeError("Correlation coefficients must be numeric")

        diff = abs(spearman_corr - pearson_corr)

        explanation = f"Difference: {diff:.3f}\n\n"
        return explanation
        
    except ValueError as ve:
        return f"Value error analyzing correlations: {ve}"
    except TypeError as te:
        return f"Type error analyzing correlations: {te}"
    except Exception as e:
        return f"Unexpected error analyzing correlations: {e}"

def perform_eda(X, y):
    """
    Perform comprehensive Exploratory Data Analysis on Boston housing dataset.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (MEDV - median house value)
        
    Returns
    -------
    dict
        Dictionary containing EDA results, insights, and recommendations
    """
    try:
        if X is None or y is None:
            raise ValueError("Input data cannot be None")
        if len(X) != len(y):
            raise ValueError("Features and target must have same length")
        if len(X) == 0:
            raise ValueError("Data cannot be empty")

        eda_results = {
            'missing_percentage': {},
            'correlations': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # 1. Missing values analysis
        try:
            for col in X.columns:
                missing_pct = (X[col].isnull().sum() / len(X)) * 100
                eda_results['missing_percentage'][col] = missing_pct
            
            # Calculate overall missing percentage
            total_cells = X.size + len(y)
            total_missing = X.isnull().sum().sum() + y.isnull().sum()
            total_missing_pct = (total_missing / total_cells) * 100
        except Exception as e:
            print(f"Error in missing values analysis: {e}")
            total_missing_pct = 0
        
        # 2. Correlation analysis
        try:
            X_imputed = X.copy()
            for col in X_imputed.columns:
                if X_imputed[col].isnull().any():
                    imputer = SimpleImputer(strategy='median')
                    X_imputed[col] = imputer.fit_transform(X_imputed[[col]]).ravel()
            
            # Analyze correlations with target variable
            for col in X_imputed.columns:
                try:
                    pearson_corr = calculate_pearson_correlation(X_imputed[col].values, y.values)
                    spearman_corr = calculate_spearman_correlation(X_imputed[col].values, y.values)
                    explanation = analyze_correlation_difference(spearman_corr, pearson_corr)
                    
                    # Save significant correlations
                    if abs(pearson_corr) > 0.5 or abs(spearman_corr) > 0.5:
                        eda_results['correlations'][col] = {
                            'pearson': pearson_corr,
                            'spearman': spearman_corr,
                            'explanation': explanation
                        }
                except Exception as col_error:
                    print(f"Error analyzing correlation for column {col}: {col_error}")
                    continue
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
        
        # 3. Generate key insights
        try:
            high_corr_features = [col for col, corrs in eda_results['correlations'].items() 
                                 if abs(corrs['pearson']) > 0.6]
            
            eda_results['key_insights'].extend([
                f"Dataset contains {total_missing_pct:.1f}% missing values",
                f"Features most correlated with price: {', '.join(high_corr_features[:3]) if high_corr_features else 'none detected'}",
                f"Data dimensions: {X.shape[0]} rows, {X.shape[1]} features",
                f"Target variable MEDV: mean value {y.mean():.2f}, standard deviation {y.std():.2f}"
            ])
        except Exception as e:
            print(f"Error generating key insights: {e}")
        
        # 4. Generate recommendations
        try:
            if total_missing_pct > 0:
                eda_results['recommendations'].append("Recommend imputation of missing values using median values")
            
            if high_corr_features:
                eda_results['recommendations'].append(f"Features {', '.join(high_corr_features)} have high correlation with target variable and may be important for modeling")
            
            eda_results['recommendations'].extend([
                "Recommend feature scaling before building models",
                "Check for outliers in features",
                "Consider using KNN imputation or machine learning-based methods for handling missing values"
            ])
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        
        return eda_results
        
    except ValueError as ve:
        print(f"Value error in EDA: {ve}")
        return {}
    except Exception as e:
        print(f"Unexpected error in EDA: {e}")
        return {}

def create_summarizer(model_name="facebook/bart-large-cnn"):
    """
    Create a text summarization pipeline using the specified model.
    
    Parameters
    ----------
    model_name : str
        Name of the summarization model to use
        
    Returns
    -------
    pipeline
        Text summarization pipeline object
    """
    try:
        if not isinstance(model_name, str):
            raise TypeError("Model name must be a string")
        if not model_name:
            raise ValueError("Model name cannot be empty")
            
        return pipeline('summarization', model=model_name)
        
    except ImportError as ie:
        print(f"Import error creating summarizer: {ie}")
        return None
    except ValueError as ve:
        print(f"Value error creating summarizer: {ve}")
        return None
    except Exception as e:
        print(f"Unexpected error creating summarizer: {e}")
        return None

def summarize_text(summarizer, text, max_length=50, min_length=25):
    """
    Generate a summary of the input text using the provided summarizer.
    
    Parameters
    ----------
    summarizer : pipeline
        Text summarization pipeline
    text : str
        Input text to summarize
    max_length : int
        Maximum length of the summary
    min_length : int
        Minimum length of the summary
        
    Returns
    -------
    str
        Generated summary text
    """
    try:
        if summarizer is None:
            raise ValueError("Summarizer cannot be None")
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if not isinstance(max_length, int) or not isinstance(min_length, int):
            raise TypeError("Length parameters must be integers")
        if max_length <= min_length:
            raise ValueError("Max length must be greater than min length")

        # Call pipeline to generate summary with specified parameters
        result = summarizer(text, max_length=max_length, min_length=min_length, 
                          length_penalty=2.0, num_beams=4, early_stopping=True)
        return result[0]['summary_text']
        
    except ValueError as ve:
        print(f"Value error in text summarization: {ve}")
        return "Error: Could not generate summary"
    except TypeError as te:
        print(f"Type error in text summarization: {te}")
        return "Error: Could not generate summary"
    except Exception as e:
        print(f"Unexpected error in text summarization: {e}")
        return "Error: Could not generate summary"

def generate_text_report(eda_results):
    """
    Generate a comprehensive text report based on EDA results.
    
    Parameters
    ----------
    eda_results : dict
        Dictionary containing EDA analysis results
        
    Returns
    -------
    tuple
        Full report text and summarized version
    """
    try:
        if not eda_results:
            raise ValueError("EDA results cannot be empty")
        if not isinstance(eda_results, dict):
            raise TypeError("EDA results must be a dictionary")

        report = f"EDA ANALYSIS REPORT - BOSTON HOUSING\n"
        report += "=" * 50 + "\n\n"
        
        # Key insights section
        try:
            report += "KEY INSIGHTS:\n"
            for i, insight in enumerate(eda_results.get('key_insights', []), 1):
                report += f"{i}. {insight}\n"
        except Exception as e:
            print(f"Error generating key insights section: {e}")
            report += "KEY INSIGHTS: Error generating insights\n"
        
        # Missing values section
        try:
            report += "\nMISSING VALUES ANALYSIS:\n"
            missing_data = eda_results.get('missing_percentage', {})
            if missing_data:
                for col, missing_pct in missing_data.items():
                    if missing_pct > 0:
                        report += f"- {col}: {missing_pct:.1f}% missing\n"
            else:
                report += "No missing values data available\n"
        except Exception as e:
            print(f"Error generating missing values section: {e}")
            report += "\nMISSING VALUES ANALYSIS: Error generating analysis\n"
        
        # Correlations section
        try:
            report += "\nSIGNIFICANT CORRELATIONS WITH TARGET VARIABLE (MEDV):\n"
            correlations = eda_results.get('correlations', {})
            if correlations:
                for col, corrs in correlations.items():
                    report += f"- {col}: Pearson={corrs.get('pearson', 'N/A'):.3f}, Spearman={corrs.get('spearman', 'N/A'):.3f}\n"
            else:
                report += "No significant correlations found\n"
        except Exception as e:
            print(f"Error generating correlations section: {e}")
            report += "\nCORRELATIONS: Error generating correlation analysis\n"
        
        # Recommendations section
        try:
            report += "\nRECOMMENDATIONS:\n"
            recommendations = eda_results.get('recommendations', [])
            if recommendations:
                for i, recommendation in enumerate(recommendations, 1):
                    report += f"{i}. {recommendation}\n"
            else:
                report += "No recommendations available\n"
        except Exception as e:
            print(f"Error generating recommendations section: {e}")
            report += "\nRECOMMENDATIONS: Error generating recommendations\n"
        
        # Generate summary
        try:
            summarizer = create_summarizer()
            if summarizer:
                summary = summarize_text(summarizer, report, max_length=60, min_length=30)
            else:
                summary = "Summary unavailable: summarizer not available"
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Summary unavailable due to error"
        
        return summary
        
    except ValueError as ve:
        error_msg = f"Value error generating report: {ve}"
        return error_msg, "Summary unavailable"
    except TypeError as te:
        error_msg = f"Type error generating report: {te}"
        return error_msg, "Summary unavailable"
    except Exception as e:
        error_msg = f"Unexpected error generating report: {e}"
        return error_msg, "Summary unavailable"

def save_to_html(summary_text, filename):
    """
    Save the EDA report and summary to an HTML file.
    
    Parameters
    ----------
    report : str
        Full EDA report text
    summary_text : str
        Summarized version of the report
    filename : str
        Output HTML filename
    """
    try:
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string")
        if not filename.endswith('.html'):
            raise ValueError("Filename must have .html extension")
        
        formatted_summary = summary_text.replace('\n', '<br>')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EDA Report - Boston Housing</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 40px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #2c3e50;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin: 0;
                    font-size: 2.2em;
                }}
                .content {{
                    font-size: 16px;
                    line-height: 1.8;
                }}
                .content br {{
                    margin-bottom: 10px;
                    display: block;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="content">
                    {formatted_summary}
                </div>

                <div class="footer">
                    <p>Generated automatically | EDA Analysis System</p>
                </div>
            </div>
        </body>
        </html>
        """
        # Save HTML file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report successfully saved to {filename}")
        
    except ValueError as ve:
        print(f"Value error saving HTML: {ve}")
    except TypeError as te:
        print(f"Type error saving HTML: {te}")
    except IOError as ioe:
        print(f"IO error saving HTML file: {ioe}")
    except Exception as e:
        print(f"Unexpected error saving HTML: {e}")

def main():
    """
    Main execution function for Boston housing EDA analysis.
    Loads data, performs EDA, generates report, and saves to HTML.
    """
    try:
        X, y = load_boston_data()
        
        if X is None or y is None:
            raise ValueError("Failed to load Boston housing dataset")

        eda_results = perform_eda(X, y)

        if not eda_results:
            raise ValueError("EDA analysis failed to produce results")

        text = generate_text_report(eda_results)
        
        save_to_html(text, "boston_eda_report.html")
        
        print("EDA analysis completed successfully!")
        
    except ValueError as ve:
        print(f"Value error in main execution: {ve}")
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")

if __name__ == "__main__":
    main()