import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ingest import process_pdf, get_collection
from rag import generate_answer


# Load environment variables
load_dotenv()


# Flask application
app = Flask(__name__)

# Maximum upload size: 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# Upload configuration
UPLOAD_FOLDER = "data/uploads"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Helper: Check allowed file type
# ---------------------------------------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({
        "status": "success",
        "message": "HomeFix AI is running"
    })


# ---------------------------------------------------------
# Open Uploaded PDF
# ---------------------------------------------------------
@app.route("/documents/<path:filename>")
def open_document(filename):

    # 1. Try exact filename
    exact_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if os.path.isfile(exact_path):
        return send_from_directory(
            UPLOAD_FOLDER,
            filename,
            as_attachment=False
        )

    # 2. Try Flask-safe filename
    # Example:
    # Warranty (1).pdf
    # becomes:
    # Warranty_1.pdf
    safe_filename = secure_filename(filename)

    safe_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    if os.path.isfile(safe_path):
        return send_from_directory(
            UPLOAD_FOLDER,
            safe_filename,
            as_attachment=False
        )

    # 3. Search files with UUID prefix
    # Example stored file:
    # a82f91cd_Warranty_1.pdf
    for stored_filename in os.listdir(UPLOAD_FOLDER):

        # Direct match
        if stored_filename == filename:
            return send_from_directory(
                UPLOAD_FOLDER,
                stored_filename,
                as_attachment=False
            )

        # Match secure filename
        if stored_filename == safe_filename:
            return send_from_directory(
                UPLOAD_FOLDER,
                stored_filename,
                as_attachment=False
            )

        # Match UUID-prefixed file
        if stored_filename.endswith(
            "_" + safe_filename
        ):
            return send_from_directory(
                UPLOAD_FOLDER,
                stored_filename,
                as_attachment=False
            )

        # Also try original filename
        if stored_filename.endswith(
            "_" + filename
        ):
            return send_from_directory(
                UPLOAD_FOLDER,
                stored_filename,
                as_attachment=False
            )

    # PDF not found
    return jsonify({
        "success": False,
        "message": f"PDF not found: {filename}"
    }), 404


# ---------------------------------------------------------
# Upload PDF Documents
# ---------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_documents():

    if "files" not in request.files:
        return jsonify({
            "success": False,
            "message": "No PDF files were selected."
        }), 400

    files = request.files.getlist("files")

    if not files:
        return jsonify({
            "success": False,
            "message": "Please select at least one PDF."
        }), 400

    uploaded_files = []

    # Get ChromaDB collection
    collection = get_collection(reset=False)

    for file in files:

        # Skip empty file
        if not file or not file.filename:
            continue

        # Check PDF
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": (
                    f"{file.filename} is not a PDF file."
                )
            }), 400

        # Convert filename to safe filename
        original_name = secure_filename(
            file.filename
        )

        # Add unique ID to avoid filename conflicts
        unique_name = (
            f"{uuid.uuid4().hex[:8]}_"
            f"{original_name}"
        )

        # Final file path
        file_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        # Save uploaded PDF
        file.save(file_path)

        try:

            # Extract, chunk and store in ChromaDB
            process_pdf(
                file_path,
                collection
            )

            # Return original display filename
            uploaded_files.append({
                "filename": file.filename,
                "stored_filename": unique_name
            })

        except Exception as error:

            # Remove failed upload
            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "success": False,
                "message": (
                    f"Could not process "
                    f"{file.filename}: {str(error)}"
                )
            }), 500

    # No valid files
    if not uploaded_files:
        return jsonify({
            "success": False,
            "message": "No valid PDF files were uploaded."
        }), 400

    # Success response
    return jsonify({
        "success": True,
        "message": (
            f"{len(uploaded_files)} PDF(s) "
            "uploaded and indexed successfully."
        ),
        "files": uploaded_files
    })


# ---------------------------------------------------------
# Chat API
# ---------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "Invalid request."
        }), 400

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:
        return jsonify({
            "success": False,
            "message": "Please enter a question."
        }), 400

    try:

        # Generate RAG answer
        result = generate_answer(question)

        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        })

    except Exception as error:

        print(
            f"Chat error: {error}"
        )

        return jsonify({
            "success": False,
            "message": (
                "Something went wrong while "
                "processing your question."
            )
        }), 500


# ---------------------------------------------------------
# Run Flask Application
# ---------------------------------------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", 5000)
        ),
        debug=True
    )