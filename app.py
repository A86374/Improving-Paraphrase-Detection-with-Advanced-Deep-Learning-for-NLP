import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import Flask, render_template, request, send_file, flash
from sklearn.exceptions import NotFittedError

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Load the saved model and vectorizer
try:
    rf_model = joblib.load('random_forest_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    rf_model = None
    tfidf_vectorizer = None
    print("Model or vectorizer not found. Please ensure 'random_forest_model.pkl' and 'tfidf_vectorizer.pkl' are present.")

# Function to generate PDF report
def generate_pdf_report(results, filename="similarity_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Similarity Report")
    c.drawString(100, 725, "----------------------------------------")

    y_position = 700
    for i, result in enumerate(results):
        c.drawString(100, y_position, f"Sentence Pair {i + 1}:")
        c.drawString(120, y_position - 15, f"Sentence 1: {result['Sentence 1']}")
        c.drawString(120, y_position - 30, f"Sentence 2: {result['Sentence 2']}")
        c.drawString(120, y_position - 45, f"Paraphrase Prediction: {result['Paraphrase Prediction']}")
        c.drawString(120, y_position - 60, f"Similarity Score: {result['Similarity Score']:.2f}")
        y_position -= 80  # Move down for the next pair
        if y_position < 50:  # Create a new page if the current one is filled
            c.showPage()
            y_position = 750  # Reset y_position for the new page

    c.save()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get sentence pairs from form input
        sentence1_list = request.form.getlist('sentence1[]')
        sentence2_list = request.form.getlist('sentence2[]')

        # Validate the input format and ensure each line has a pair of sentences
        new_sentence_pairs = []
        invalid_lines = []

        for i, (s1, s2) in enumerate(zip(sentence1_list, sentence2_list)):
            s1 = s1.strip()
            s2 = s2.strip()
            if s1 and s2:
                new_sentence_pairs.append((s1, s2))
            else:
                invalid_lines.append((i + 1, s1, s2))

        # Show error if no valid sentence pairs found
        if not new_sentence_pairs:
            flash("Error: No valid sentence pairs found. Please ensure each sentence pair is filled in.", "danger")
            return render_template('upload.html')

        # If there are invalid lines, display them
        if invalid_lines:
            error_message = "The following lines do not match the expected format:\n"
            for line_num, s1, s2 in invalid_lines:
                error_message += f"Line {line_num}: Sentence 1: '{s1}' | Sentence 2: '{s2}'\n"
            flash(error_message, "danger")
            return render_template('upload.html')

        # Combine sentence pairs into a single feature for prediction
        new_combined_sentences = [f"{pair[0]} {pair[1]}" for pair in new_sentence_pairs]

        try:
            # Convert new sentences to TF-IDF features
            new_sentences_tfidf = tfidf_vectorizer.transform(new_combined_sentences)

            # Predict using the loaded model
            new_predictions = rf_model.predict(new_sentences_tfidf)

            # Calculate similarity scores (Cosine Similarity)
            similarity_scores = cosine_similarity(new_sentences_tfidf)

            # Prepare the results for export
            results = []
            for i, pair in enumerate(new_sentence_pairs):
                results.append({
                    "Sentence 1": pair[0],
                    "Sentence 2": pair[1],
                    "Paraphrase Prediction": "Yes" if new_predictions[i] == 1 else "No",
                    "Similarity Score": np.max(similarity_scores[i])  # Get the maximum similarity score
                })

            # Create a DataFrame from the results
            results_df = pd.DataFrame(results)

            # Save results to CSV and Excel
            csv_file = "similarity_report.csv"
            excel_file = "similarity_report.xlsx"
            results_df.to_csv(csv_file, index=False)
            results_df.to_excel(excel_file, index=False)

            # Generate the PDF report
            pdf_file = "similarity_report.pdf"
            generate_pdf_report(results, pdf_file)

            flash("Report generated successfully!", "success")
            return render_template('upload.html', results=results, csv_file=csv_file, excel_file=excel_file, pdf_file=pdf_file)

        except NotFittedError:
            flash("Error: The TF-IDF vectorizer or model is not properly fitted. Please check your model and vectorizer.", "danger")
        except ValueError as ve:
            flash(f"Value Error: {ve}", "danger")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        flash("Error: The requested file does not exist.", "danger")
        return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get sentence pairs from form input
#         sentence1_list = request.form.getlist('sentence1[]')
#         sentence2_list = request.form.getlist('sentence2[]')

#         # Validate the input format and ensure each line has a pair of sentences
#         new_sentence_pairs = []
#         invalid_lines = []

#         for i, (s1, s2) in enumerate(zip(sentence1_list, sentence2_list)):
#             s1 = s1.strip()
#             s2 = s2.strip()
#             if s1 and s2:
#                 new_sentence_pairs.append((s1, s2))
#             else:
#                 invalid_lines.append((i + 1, s1, s2))

#         # Show error if no valid sentence pairs found
#         if not new_sentence_pairs:
#             flash("Error: No valid sentence pairs found. Please ensure each sentence pair is filled in.", "danger")
#             return render_template('upload.html')

#         # If there are invalid lines, display them
#         if invalid_lines:
#             error_message = "The following lines do not match the expected format:\n"
#             for line_num, s1, s2 in invalid_lines:
#                 error_message += f"Line {line_num}: Sentence 1: '{s1}' | Sentence 2: '{s2}'\n"
#             flash(error_message, "danger")
#             return render_template('upload.html')

#         # Combine sentence pairs into a single feature for prediction
#         new_combined_sentences = [f"{pair[0]} {pair[1]}" for pair in new_sentence_pairs]

#         try:
#             # Convert new sentences to TF-IDF features
#             new_sentences_tfidf = tfidf_vectorizer.transform(new_combined_sentences)

#             # Predict using the loaded model
#             new_predictions = rf_model.predict(new_sentences_tfidf)

#             # Calculate similarity scores (Cosine Similarity)
#             similarity_scores = cosine_similarity(new_sentences_tfidf)

#             # Prepare the results for export
#             results = []
#             for i, pair in enumerate(new_sentence_pairs):
#                 results.append({
#                     "Sentence 1": pair[0],
#                     "Sentence 2": pair[1],
#                     "Paraphrase Prediction": "Yes" if new_predictions[i] == 1 else "No",
#                     "Similarity Score": np.max(similarity_scores[i])  # Get the maximum similarity score
#                 })

#             # Create a DataFrame from the results
#             results_df = pd.DataFrame(results)

#             # Save results to CSV and Excel
#             csv_file = "similarity_report.csv"
#             excel_file = "similarity_report.xlsx"
#             results_df.to_csv(csv_file, index=False)
#             results_df.to_excel(excel_file, index=False)

#             # Generate the PDF report
#             pdf_file = "similarity_report.pdf"
#             generate_pdf_report(results, pdf_file)

#             flash("Report generated successfully!", "success")
#             return render_template('upload.html', results=results, csv_file=csv_file, excel_file=excel_file, pdf_file=pdf_file)

#         except NotFittedError:
#             flash("Error: The TF-IDF vectorizer or model is not properly fitted. Please check your model and vectorizer.", "danger")
#         except ValueError as ve:
#             flash(f"Value Error: {ve}", "danger")
#         except Exception as e:
#             flash(f"An unexpected error occurred: {e}", "danger")

#     return render_template('upload.html')

# @app.route('/download/<filename>')
# def download_file(filename):
#     try:
#         return send_file(filename, as_attachment=True)
#     except FileNotFoundError:
#         flash("Error: The requested file does not exist.", "danger")
#         return render_template('upload.html')

# if __name__ == "__main__":
#     app.run(debug=True)
