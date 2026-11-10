import os
import subprocess
import markdown

with open("experiments_report.md", "r", encoding="utf-8") as f:
    text = f.read()

body = markdown.markdown(text, extensions=["tables", "fenced_code"])

html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ASTE Experimental Report</title>
<style>
  @page {
    size: A4;
    margin: 18mm 15mm 18mm 15mm;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #24292f;
    line-height: 1.55;
    font-size: 10.5pt;
  }
  h1 {
    font-size: 18pt;
    border-bottom: 2px solid #0969da;
    padding-bottom: 6px;
    margin-top: 0;
    color: #0969da;
  }
  h2 {
    font-size: 13pt;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 4px;
    margin-top: 22px;
    color: #1f2328;
  }
  h3 {
    font-size: 11pt;
    margin-top: 14px;
    color: #24292f;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 16px 0;
    font-size: 9pt;
  }
  th, td {
    border: 1px solid #d0d7de;
    padding: 6px 10px;
    text-align: left;
  }
  th {
    background-color: #f6f8fa;
    font-weight: 600;
  }
  tr:nth-child(even) {
    background-color: #fcfdfe;
  }
  code {
    background-color: #f6f8fa;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 85%;
    font-family: Consolas, "Liberation Mono", Menlo, monospace;
  }
  hr {
    height: 1px;
    background-color: #d0d7de;
    border: none;
    margin: 18px 0;
  }
  strong {
    color: #0969da;
  }
</style>
</head>
<body>
""" + body + """
</body>
</html>"""

html_path = os.path.abspath("experiments_report.html")
pdf_path = os.path.abspath("experiments_report.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML written to: {html_path}")

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    edge_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    html_path
]

print("Printing to PDF via headless browser...")
subprocess.run(cmd, check=True)
print(f"PDF generated successfully at: {pdf_path}")
