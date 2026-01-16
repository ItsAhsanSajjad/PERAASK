from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from retriever import retrieve
from answerer import answer_question

app = FastAPI()

# Serve static files (PDFs, DOCX, etc.)
app.mount(
    "/assets/data",
    StaticFiles(directory=os.path.join("assets", "data")),
    name="data"
)

class QueryRequest(BaseModel):
    user_id: str
    message: str

class QueryResponse(BaseModel):
    user_id: str
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    # Base HTML with tighter CSS to avoid phantom spacing
    html_answer = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">
      <p style="margin:0; padding:0;">{result['answer']}</p>
    """

    references = result.get("references", [])
    if references:
        html_answer += """
        <hr style="margin:8px 0;" />
        <h3 style="margin:4px 0;">References</h3>
        <ol style="margin:0; padding-left:18px;">
        """
        for ref in references:
            doc = ref.get("document", "Unknown document")
            path = ref.get("path", "")
            url_hint = ref.get("url_hint", "")
            snippet = ref.get("snippet", "")

            html_answer += f"""
            <li style="margin-bottom:6px;">
              <a href="http://10.53.34.67:8020/{path}{url_hint}" target="_blank" download>{doc}</a>
              <p style="margin:2px 0;">{snippet}</p>
            </li>
            """
        html_answer += "</ol>"

    html_answer = html_answer.rstrip("</p>\n    </div>") + "</p></div>"

    return QueryResponse(user_id=request.user_id, answer=html_answer.strip())
