import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import chardet
import time
from scipy.stats import zscore
import numpy as np
from joypy import joyplot
matplotlib.use('Agg')

# Set OpenAI API Token from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("Error: OpenAI API token is not set.")
    sys.exit(1)

def load_dataset(filename, fallback_encoding="ISO-8859-1"):

    print(f"Step 1: Loading dataset from {filename}...")
    try:
        # Detect encoding
        with open(filename, 'rb') as f:
            result = chardet.detect(f.read())
            detected_encoding = result['encoding']
            print(f"    - Detected encoding: {detected_encoding if detected_encoding else 'unknown'}")

        # Attempt to load the dataset
        encoding_to_use = detected_encoding if detected_encoding else fallback_encoding
        df = pd.read_csv(filename, encoding=encoding_to_use, on_bad_lines='skip')
        print("     - Dataset loaded successfully!")
        return df

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please check the path and try again.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filename}' is empty or contains no valid data.")
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the file '{filename}'. {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
    
    # Stop execution if any exception occurs
    print("Stopping program execution due to the error.")
    sys.exit(1)


def preliminary_analysis(df, output_folder="./"):
    summary = {}
    correlation_heatmap_path = None
    print("Step 2: Preliminary Analysis")
    # Compute missing values
    print("     -Computing missing values...")
    missing_values = df.isnull().sum()
    summary['missing_values'] = missing_values

    # Compute correlation matrix
    print("     -Computing correlation matrix...")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    if numeric_df.empty:
        correlation_matrix = pd.DataFrame()  # Handle case where no numeric columns exist
        print("No numeric columns available for correlation matrix.")
    else:
        correlation_matrix = numeric_df.corr()
        # Save correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        correlation_heatmap_path = f"{output_folder}/correlation_heatmap.png"
        plt.savefig(correlation_heatmap_path)
        plt.close()
    
    summary['correlation_matrix'] = correlation_matrix

    # Detect outliers using Z-score method
    print("     -Detecting outliers...")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_counts = {}
    for col in numerical_columns:
        z_scores = zscore(df[col].dropna())
        outliers = (abs(z_scores) > 3).sum()
        outlier_counts[col] = outliers
    summary['outlier_counts'] = outlier_counts

    # Convert summary to text format
    summary_text = "\n".join([
        "Missing Values:",
        missing_values.to_string(),
        "\nCorrelation Matrix:",
        correlation_matrix.to_string() if not correlation_matrix.empty else "No numeric data available for correlation.",
        "\nOutlier Counts:",
        "\n".join([f"{col}: {count}" for col, count in outlier_counts.items()])
    ])
    print("     -Summary analysis completed.")

    return summary, summary_text, correlation_heatmap_path

def get_dataset_preview(df, rows=30):
    print(f"    -Extracting first {rows} rows of the dataset for preview...")
    try:
        preview = df.head(rows).to_string(index=False)
        return preview
    except Exception as e:
        print(f"Error extracting dataset preview: {e}")
        return "Unable to load dataset preview."

def get_chatgpt_response(prompt, role="user", max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": role, "content": prompt}
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error from ChatGPT (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print("Retrying ChatGPT request...")
                time.sleep(2)
            else:
                print("Failed to get a valid response from ChatGPT after multiple attempts.")
                sys.exit(1)

def suggest_graphs(df, dataset_preview, summary_text):
    print("Step 3: Getting graph suggestions from ChatGPT...")
    import numpy as _np
    cols = df.columns.tolist()
    nrows, ncols = df.shape
    numeric = df.select_dtypes(include=[_np.number]).columns.tolist()
    categorical = [c for c in cols if c not in numeric]
    # crude datetime detection without changing dtypes
    datetime_like = [c for c in cols if any(k in c.lower() for k in ["date","time","timestamp","year","month","day"])]
    card = {c: int(df[c].nunique(dropna=True)) for c in categorical[:20]}  # cap to keep prompt small

    prompt = f"""
    You are choosing 2 *different* visualization types from: bar plot, pie chart, scatterplot, histogram with density curve.
    Use ONLY existing columns: {', '.join(cols)}. Rows={nrows}, Cols={ncols}.

    ### Data profile (for guidance)
    - numeric columns: {', '.join(numeric) if numeric else 'none'}
    - categorical columns (with cardinality): {card}
    - datetime-like columns (name heuristics): {datetime_like}

    ### STRICT selection rules
    1) **Scatterplot**: pick when you can plot two numeric variables (X,Y). Prefer when nrows ≥ 200. Avoid if either axis has < 10 unique values.
    2) **Histogram with density**: pick for a single numeric variable to show distribution. Prefer when the variable is continuous or has ≥ 20 unique values.
    3) **Bar plot**: pick when you have 1 categorical (≤ 20 categories) and 1 numeric; aggregate with mean/median if many rows. Never choose bar if category count > 20.
    4) **Pie chart**: allowed only when a single categorical has **≤ 6** categories **and** the values are mutually exclusive & sum to a meaningful whole (e.g., shares). Otherwise DO NOT suggest pie.
    5) If any datetime-like column exists and you need “trend over time”, choose **bar** with the datetime column aggregated by month/quarter (line plots are not allowed in this task).
    6) Variables must exist exactly as named. Prefer numeric variables with wider variance.
    7) Ensure the two suggestions yield **non-redundant** insights (e.g., not two distributions of nearly identical fields).

    ### OUTPUT (exactly two blocks, no prose)
    Graph Type: <bar plot | pie chart | scatterplot | histogram with density curve>
    Variables: <comma-separated column names used, usually 1 for hist, 2 for others>
    """
    response = get_chatgpt_response(prompt)
    return response

def remove_outliers(df, columns):
    # Remove outliers from specified columns in the dataframe using the IQR method.
    print("Removing outliers...")
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:  # Only process numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def create_graphs(df, suggestions, output_folder):
    dfscatter = df
    # Generate multiple graphs based on ChatGPT's suggestions and save them as PNG files.
    print("Step 4: Generating graphs based on suggestions...")
    df.columns = df.columns.str.strip()  # Ensure column names are clean
    graph_paths = []

    # Limit data size if too large
    if len(df) > 10000:  # Adjust the threshold as needed
        print(f"Dataset is too large ({len(df)} rows). Sampling down to 10,000 rows for visualization.")
        df = df.sample(n=10000)
    
    exception_details = []  # To collect detailed exception logs

    for suggestion in suggestions.strip().split('\n\n'):
        retry_count = 0
        while retry_count < 3:  # Retry a maximum of 3 times for each suggestion
            try:
                lines = [line.strip() for line in suggestion.splitlines() if line.strip()]
                if len(lines) < 2:
                    print(f"Skipping invalid suggestion block:\n{suggestion}")
                    break  # Move to the next suggestion

                # Parse graph type and variables
                graph_type_line = lines[0]
                variable_line = lines[1]

                if ':' not in graph_type_line or ':' not in variable_line:
                    print(f"Skipping invalid suggestion block:\n{suggestion}")
                    break

                _, gtype = graph_type_line.split(':', 1)
                graph_type = gtype.strip().lower()
                print("graph_type: ", graph_type)
                _, vars_line = variable_line.split(':', 1)
                variables = [v.strip() for v in vars_line.split(',')]

                # Ensure all variables exist in the dataframe
                if not all(var in df.columns for var in variables):
                    print(f"One or more variables not found in dataset columns: {variables}")
                    break
                
                # Create figure
                plt.figure(figsize=(10, 6))


                if "scatterplot" in graph_type:
                    # Remove outliers from all numeric columns before plotting
                    numeric_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
                    df = remove_outliers(df, numeric_columns)

                    # Handle large axis ranges by applying normalization or log scaling
                    if len(variables) > 1:
                        for var in variables[:2]:  # Only consider the first two variables for axis ranges
                            max_val, min_val = df[var].max(), df[var].min()
                            range_val = max_val - min_val
                            if range_val > 1e6:
                                print(f"Large axis range detected for {var}. Applying log transformation.")
                                df[var] = np.log1p(df[var])  # Log-transform the data

                    sns.scatterplot(data=df, x=variables[0], y=variables[1])
                    plt.title(f"Scatterplot of {variables[0]} vs {variables[1]}")

                elif "bar plot" in graph_type:
                    sns.barplot(data=df, x=variables[0], y=variables[1], errorbar=None)
                    plt.title(f"Bar Plot of {variables[0]} vs {variables[1]}")
                    for i, val in enumerate(df[variables[1]]):
                        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')
                    plt.xlabel(variables[0])
                    plt.ylabel(variables[1])

                elif "histogram with density curve" in graph_type:
                    sns.histplot(data=df, x=variables[0], kde=True)
                    plt.title(f"Histogram with Density Curve of {variables[0]}")
                    plt.xlabel(variables[0])
                    plt.ylabel("Frequency")

                elif "line plot" in graph_type:
                    plt.plot(df[variables[0]], df[variables[1]], marker='o')
                    plt.title(f"Line Plot of {variables[0]} vs {variables[1]}")
                    plt.xlabel(variables[0])
                    plt.ylabel(variables[1])


                elif "pie chart" in graph_type:
                    df[variables[0]].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
                    plt.title(f"Pie Chart of {variables[0]}")
                    plt.ylabel("")  # Remove default y-axis label

                else:
                    print(f"Unrecognized graph type '{graph_type}'. Skipping.")
                    break


                # Save the graph
                graph_filename = f"{graph_type.replace(' ', '_')}.png"
                graph_path = os.path.join(output_folder, graph_filename)
                plt.savefig(graph_path, dpi=100, bbox_inches='tight')  # Ensure DPI = 100
                plt.close()
                graph_paths.append(graph_path)
                print(f"    - Saved {graph_type} as {graph_path}")
                break # Exit retry loop after successful graph creation

            except Exception as e:
                error_message = f"Error creating graph (attempt {retry_count + 1}): {e}"
                print(error_message)
                exception_details.append({
                    "suggestion": suggestion,
                    "attempt": retry_count + 1,
                    "error": str(e)
                })
                retry_count += 1
                if retry_count >= 3:
                    print("Failed to create graph after 3 attempts. Moving to the next suggestion.")

    # Log all exceptions
    if exception_details:
        print("\nDetailed Exception Log:")
        for detail in exception_details:
            print(f"Suggestion: {detail['suggestion']}")
            print(f"Attempt: {detail['attempt']}")
            print(f"Error: {detail['error']}\n")

    return graph_paths

def create_readme(dataset_preview, graph_paths, summary_text, correlation_heatmap_path, output_folder):
    print("Step 5: Generating README.md file...")
    time.sleep(5)

    graph_details = []
    graph_markdown = []
    for graph_path in graph_paths:
        graph_filename = os.path.basename(graph_path)
        graph_type = os.path.splitext(graph_filename)[0].replace('_', ' ').title()
        graph_details.append(f"- Graph Type: {graph_type}, File: {graph_filename}")
        graph_markdown.append(f"### {graph_type}\n![{graph_type}]({graph_filename})\n")

    # Ensure the correlation heatmap also gets its own insights section
    if correlation_heatmap_path:
        heatmap_file = os.path.basename(correlation_heatmap_path)
        graph_details.append(f"- Graph Type: Correlation Matrix Heatmap, File: {heatmap_file}")

    prompt = f"""
    You are a senior data analyst. Produce *decision-grade* insights (not generic summaries). You MUST write a subsection for **each** of these figures, using the exact headings shown: {', '.join(graph_details)}.

    ## INPUT
    - Dataset Preview (first rows): ```\\n{dataset_preview}\\n```
    - Summary Analysis (missing values, outliers, correlations): ```\\n{summary_text}\\n```
    - Generated Graphs: {', '.join(graph_details)}

    ## OUTPUT (Markdown only; no images and no raw matrices)
    # <Concise, specific title>
    ## Introduction
    - 2-3 sentences on the business/stakeholder goal this analysis enables.

    ## Preliminary Analysis
    - Bullets for **missing values** and **outliers** (no raw dumps).
    - **Correlation**: If >12 numeric columns, output a 3-column table with the **top 10 absolute correlations** only:
      `| feature_a | feature_b | corr |` (round to 3 decimals). Never paste full matrices.

    ## Visualization Insights
    - For **each generated figure listed above**, add `### <Figure name>` (use the exact name) followed by **at least 5 non-obvious, actionable bullets**.
    - Each bullet should be decision-oriented: e.g., thresholds, cohorts to watch, segments to target, hypotheses to test, anomalies to validate, features to engineer, or metrics to track.
    - When applicable, suggest a simple rule or alert (e.g., “if X > 1.2× median and Y < 0.3, flag for review”).
    - Refer to column names that actually exist.

    ## Key Insights
    - 4-7 bullets summarizing takeaways a PM/analyst could act on this week.

    ## Implications
    - Concrete next steps with owners where possible (Data/Ops/Eng/Marketing).

    ## Conclusion
    - 2-3 sentences wrap-up with risks/assumptions.
    """

    try:
        readme_content = get_chatgpt_response(prompt)
        readme_full_content = (
            readme_content
            + "\n\n<details><summary><strong>Preliminary Test Results</strong></summary>\n\n```text\n"
            + summary_text
            + "\n```\n</details>\n"
        )

        readme_path = os.path.join(output_folder, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_full_content)
        print(f"    -README.md file created at {readme_path}")

    except Exception as e:
        print(f"Error generating README: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py dataset.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_prefix = os.path.splitext(os.path.basename(input_file))[0]

    output_folder = output_prefix
    os.makedirs(output_folder, exist_ok=True)

    df = load_dataset(input_file)

    # Limit dataset rows globally
    if len(df) > 100000:
        print(f"Dataset is too large ({len(df)} rows). Sampling down to 100,000 rows.")
        df = df.sample(n=100000)

    dataset_preview = get_dataset_preview(df)

    summary, summary_text, correlation_heatmap_path = preliminary_analysis(df, output_folder)

    suggestions = suggest_graphs(df, dataset_preview, summary_text)

    graphs = create_graphs(df, suggestions, output_folder)

    create_readme(dataset_preview, graphs, summary_text, correlation_heatmap_path, output_folder)

    print("\nProcess completed successfully!")
    print(f"Generated graphs and README saved in '{output_folder}':")
    all_graphs = graphs + ([correlation_heatmap_path] if correlation_heatmap_path else [])
    for graph in all_graphs:
        print(f" - {os.path.basename(graph)}")

if __name__ == "__main__":
    main()