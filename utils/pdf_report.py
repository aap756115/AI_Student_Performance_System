from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf(student_data, final_score, report):

    filename = "Student_Report.pdf"

    c = canvas.Canvas(filename, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, 760, "Student Performance Report")

    # Student Details
    c.setFont("Helvetica", 12)

    y = 720

    for key, value in student_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.drawString(50, y, f"Predicted Score: {final_score}")
    y -= 35

    # AI Report Heading
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "AI Performance Report")
    y -= 25

    # AI Report
    c.setFont("Helvetica", 10)

    for line in report.split("\n"):

        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 750

        c.drawString(50, y, line[:95])
        y -= 15

    c.save()

    return filename