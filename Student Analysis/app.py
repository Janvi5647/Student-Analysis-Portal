from flask import Flask, render_template, request, send_file
import matplotlib
matplotlib.use('Agg')  
import pandas as pd
import os
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = Flask(__name__)

file_path = 'Project.xlsx'
df = pd.read_excel(file_path, header=0, dtype={0: str})  

df = df.fillna(0)  

for col in df.columns:
    if 'Unnamed' in str(col):
        col_idx = df.columns.get_loc(col)
        subject_idx = (col_idx - 3) // 4  
        test_num = (col_idx - 3) % 4 + 1  
        if subject_idx < len(["Java", "Maths", "Physics", "SE"]):
            new_name = f"{['Java', 'Maths', 'Physics', 'SE'][subject_idx]} Test{test_num}"
            df = df.rename(columns={col: new_name})

df.columns = df.columns.str.strip()
print("Available columns:", list(df.columns))  
enrollment_col = df.columns[0] 
roll_col = df.columns[1]       
student_col = df.columns[2]    

df[student_col] = df[student_col].astype(str).str.strip()

subjects = []
subject_tests = {}
start_col = 3  

while start_col < len(df.columns):
    if start_col + 4 <= len(df.columns):
        subject_name = df.columns[start_col].split()[0]
        subject_name = df.columns[start_col].split()[0]
        subjects.append(subject_name)
        
        subject_tests[subject_name] = df.columns[start_col:start_col + 4]
        start_col += 4
    else:
        break

print("Detected subjects:", subjects)
print("Subject tests:", subject_tests)

for subject, columns in subject_tests.items():
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0)

df["Total Marks"] = df.iloc[:, 3:].fillna(0).sum(axis=1)

def get_grade(marks, max_marks=100):
    percentage = (marks/max_marks) * 100
    if percentage >= 90: return 'A+'
    elif percentage >= 80: return 'A'
    elif percentage >= 70: return 'B+'
    elif percentage >= 60: return 'B'
    elif percentage >= 50: return 'C'
    else: return 'F'

def calculate_student_performance(enrollment_no):
    df[enrollment_col] = df[enrollment_col].astype(str).str.strip()
    enrollment_no = str(enrollment_no).strip()

    student_data = df[df[enrollment_col] == enrollment_no]  
    
    if student_data.empty:
        return None
    
    subject_performance = {}
    for subject, tests in subject_tests.items():
        test_marks = student_data[tests].iloc[0].fillna(0)  
        total_marks = test_marks.sum()
        avg_marks = total_marks / 4
        
        sorted_marks = sorted(test_marks)
        initial_performance = sum(sorted_marks[:2]) / 2
        final_performance = sum(sorted_marks[-2:]) / 2
        improvement = final_performance - initial_performance
        
        subject_performance[subject] = {
            'test_marks': test_marks.to_dict(),
            'total': total_marks,
            'average': avg_marks,
            'improvement': improvement,
            'grade': get_grade(avg_marks, max_marks=25)
        }
    
    subject_averages = {subject: data['average'] for subject, data in subject_performance.items()}
    strongest_subject = max(subject_averages, key=subject_averages.get)
    weakest_subject = min(subject_averages, key=subject_averages.get)
    
    subject_improvements = {subject: data['improvement'] for subject, data in subject_performance.items()}
    most_improved_subject = max(subject_improvements, key=subject_improvements.get)
    least_improved_subject = min(subject_improvements, key=subject_improvements.get)
    
    return {
        'student_info': student_data.iloc[0].to_dict(),
        'subject_performance': subject_performance,
        'strongest_subject': strongest_subject,
        'weakest_subject': weakest_subject,
        'most_improved_subject': most_improved_subject,
        'least_improved_subject': least_improved_subject,
        'total_marks': student_data['Total Marks'].iloc[0],
        'overall_grade': get_grade(student_data['Total Marks'].iloc[0], max_marks=400)
    }

def analyze_student_performance(student_data):
    analysis = {
        'overall': {},
        'subject_wise': {},
        'recommendations': []
    }
    
    total_marks = student_data['total_marks']
    
    lower_rank_students = len(df[df["Total Marks"] < total_marks])

    total_students = len(df) - 1  

    if total_students > 0:
        percentile = (lower_rank_students / total_students) * 100
    else:
        percentile = 100  
    
    analysis['overall'] = {
        'percentile': (
            round(percentile, 2)
        ),
        'class_rank': (
            len(df[df['Total Marks'] > total_marks]) + 1
        ),
        'performance_trend': 'Improving' if sum(
            student_data['subject_performance'][subject]['improvement'] 
            for subject in subjects
        ) > 0 else 'Declining'
    }

    for subject, perf in student_data['subject_performance'].items():
        test_marks = list(perf['test_marks'].values())
        subject_avg = df[subject_tests[subject]].mean().mean()
        min_marks = min(test_marks)  
        max_marks = max(test_marks)  

        # class_marks = df[subject_tests[subject]].replace(0, np.nan)  
        # class_min = class_marks.min().min()  
        # class_max = df[subject_tests[subject]].max().max()

        lower_rank_students = len(df[df[subject_tests[subject]].mean(axis=1) < perf['average']])
        total_students = len(df) - 1  

        if total_students > 0:
            subject_percentile = (lower_rank_students / total_students) * 100
        else:
            subject_percentile = 100  # If only one student, they rank highest

        analysis['subject_wise'][subject] = {
            'performance_level': (
                'Above Average' if perf['average'] > subject_avg
                else 'Below Average' if perf['average'] < subject_avg
                else 'Average'
            ),
            'trend': 'Improving' if perf['improvement'] > 0 else 'Declining',
            'consistency': test_marks[-1] >= test_marks[-2],
            'gap_from_avg': perf['average'] - subject_avg,
            'percentile': round(subject_percentile, 2),  # Rounded to 2 decimal places
            'min_marks': min_marks,
            'max_marks': max_marks,
            'min_test': list(perf['test_marks'].keys())[list(perf['test_marks'].values()).index(min_marks)],
            'max_test': list(perf['test_marks'].keys())[list(perf['test_marks'].values()).index(max_marks)],
            # 'class_min': class_min,  # Updated class minimum (excluding zeros)
            # 'class_max': class_max
        }

    
    if analysis['overall']['performance_trend'] == 'Declining':
        analysis['recommendations'].append(
            "Focus on maintaining consistent study schedule across all subjects"
        )
    
    for subject, analysis_data in analysis['subject_wise'].items():
        if analysis_data['performance_level'] == 'Below Average':
            analysis['recommendations'].append(
                f"Need additional focus on {subject}. Consider extra practice and guidance"
            )
        if not analysis_data['consistency']:
            analysis['recommendations'].append(
                f"Work on maintaining consistency in {subject}"
            )
        if analysis_data['gap_from_avg'] < -5:  
            analysis['recommendations'].append(
                f"Urgent attention needed in {subject}. Schedule extra sessions"
            )
    
    return analysis

def generate_marksheet(student_data):
    """Generate a well-formatted PDF marksheet for a student."""
    os.makedirs("Marksheet", exist_ok=True)

    student_name = student_data['student_info'][student_col].replace(' ', '_')
    pdf_path = f"Marksheet/{student_name}_marksheet.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title Section
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=1,  # Center align
        textColor=colors.darkblue
    )
    elements.append(Paragraph("Student Marksheet", title_style))
    elements.append(Spacer(1, 12))

    student_info_data = [
        ["Enrollment No:", student_data['student_info'][enrollment_col]],
        ["Roll No:", student_data['student_info'][roll_col]],
        ["Name:", student_data['student_info'][student_col]],
        ["Total Marks:", f"{student_data['total_marks']} / 400"],
        ["Overall Grade:", student_data['overall_grade']]
    ]
    
    info_table = Table(student_info_data, colWidths=[120, 300])
    info_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 12))

    headers = ['Subject', 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Total', 'Grade']
    table_data = [headers]

    for subject, data in student_data['subject_performance'].items():
        row = [
            subject,
            *[data['test_marks'][test] for test in sorted(data['test_marks'].keys())],
            data['total'],
            data['grade']
        ]
        table_data.append(row)

    marks_table = Table(table_data, colWidths=[80, 50, 50, 50, 50, 60, 60])
    marks_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow)
    ]))
    elements.append(marks_table)
    elements.append(Spacer(1, 12))

    analysis_data = [
        ["Performance Analysis"],
        ["Strongest Subject:", student_data['strongest_subject']],
        ["Weakest Subject:", student_data['weakest_subject']],
        ["Most Improved Subject:", student_data['most_improved_subject']],
        ["Least Improved Subject:", student_data['least_improved_subject']]
    ]

    analysis_table = Table(analysis_data, colWidths=[150, 200])
    analysis_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(analysis_table)
    elements.append(Spacer(1, 12))

    if student_data.get('recommendations'):
        elements.append(Paragraph("Recommendations:", styles['Heading2']))
        recommendation_list = [
            [Paragraph(f"• {rec}", styles['BodyText'])] for rec in student_data['recommendations']
        ]
        rec_table = Table(recommendation_list, colWidths=[400])
        rec_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0, colors.white),
            ('PADDING', (0, 0), (-1, -1), 4)
        ]))
        elements.append(rec_table)
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return pdf_path


def calculate_overall_performance():
    subject_averages = {subject: df[tests].mean().mean() for subject, tests in subject_tests.items()}
    strongest_subject = max(subject_averages, key=subject_averages.get)
    weakest_subject = min(subject_averages, key=subject_averages.get)

    df["Improvement"] = sum(df[tests[-1]] - df[tests[0]] for tests in subject_tests.values())

    most_improved = df.loc[df["Improvement"].idxmax(), student_col]
    least_improved = df.loc[df["Improvement"].idxmin(), student_col]

    highest_per_test = {}
    subject_analysis = {}
    
    for subject, tests in subject_tests.items():
        highest_per_test[subject] = {}
        
        test_marks = df[tests].replace(0, np.nan)  
        min_marks = test_marks.min().min()  
        
        subject_analysis[subject] = {
            'test_averages': {},
            'test_highest': [],
            'test_lowest': [],
            'average': df[tests].mean().mean()
        }
        
        for test in tests:
            max_scorer_idx = df[test].idxmax()
            highest_per_test[subject][test] = {
                'name': df.loc[max_scorer_idx, student_col],
                'marks': df.loc[max_scorer_idx, test]
            }
            
       
            test_avg = df[test].mean()
            test_max = df[test].max()
            test_min = df[test].replace(0, np.nan).min()  
            
            subject_analysis[subject]['test_averages'][test] = test_avg
            subject_analysis[subject]['test_highest'].append(test_max)
            subject_analysis[subject]['test_lowest'].append(test_min)

    overall_topper = df.loc[df["Total Marks"].idxmax(), student_col]

    top_students = df[[student_col, roll_col, enrollment_col, "Total Marks"]] \
        .sort_values(by="Total Marks", ascending=False) \
        .head(10)

    student_consistency = {}
    student_deterioration = {}
    
    for _, row in df.iterrows():
        student_name = row[student_col]
        student_consistency[student_name] = 0
        student_deterioration[student_name] = 0
        valid_subjects = 0
        
        for subject, tests in subject_tests.items():
            test_marks = row[tests].fillna(0)  
            if test_marks.sum() > 0:  
                mark_range = test_marks.max() - test_marks.min()
                if pd.notna(mark_range):
                    student_consistency[student_name] += mark_range
                    valid_subjects += 1
            
            early_avg = test_marks.iloc[:2].mean()
            later_avg = test_marks.iloc[2:].mean()
            deterioration = early_avg - later_avg
            student_deterioration[student_name] += deterioration if pd.notna(deterioration) else 0
        
        if valid_subjects > 0:
            student_consistency[student_name] = student_consistency[student_name] / valid_subjects
    
    student_consistency = {k: v for k, v in student_consistency.items() if v > 0}
    
    most_consistent = min(student_consistency.items(), key=lambda x: x[1])[0]
    
    most_deteriorated = max(student_deterioration.items(), key=lambda x: x[1])[0]

    return (strongest_subject, weakest_subject, most_improved, least_improved, 
            highest_per_test, overall_topper, top_students, subject_analysis,
            most_consistent, most_deteriorated)

@app.route('/')
def index():
    (strongest, weakest, most_improved, least_improved, 
     highest_per_test, overall_topper, top_students, subject_analysis,
     most_consistent, most_deteriorated) = calculate_overall_performance()

    passing_marks = 35
    subject_statistics = {}
    
    for subject in subjects:
        subject_tests_list = subject_tests[subject]
        df['Subject_Total'] = df[subject_tests_list].sum(axis=1)
        
        passing_students = df[df['Subject_Total'] >= passing_marks]
        failing_students = df[(df['Subject_Total'] < passing_marks) & (df['Subject_Total'] > 0)]
        
        subject_statistics[subject] = {
            'passing': passing_students[[student_col, 'Subject_Total']].to_dict('records'),
            'failing': failing_students[[student_col, 'Subject_Total']].to_dict('records'),
            'pass_count': len(passing_students),
            'fail_count': len(failing_students)
        }
        
        df.drop('Subject_Total', axis=1, inplace=True)
    
    sorted_df = df.sort_values('Total Marks', ascending=False)
    students_per_division = 30
    
    division_statistics = {
        'A': sorted_df.iloc[0:students_per_division][[student_col, enrollment_col, 'Total Marks']].to_dict('records'),
        'B': sorted_df.iloc[students_per_division:2*students_per_division][[student_col, enrollment_col, 'Total Marks']].to_dict('records'),
        'C': sorted_df.iloc[2*students_per_division:3*students_per_division][[student_col, enrollment_col, 'Total Marks']].to_dict('records'),
        'D': sorted_df.iloc[3*students_per_division:][[student_col, enrollment_col, 'Total Marks']].to_dict('records')
    }

    return render_template('index.html', 
                         subjects=subjects,
                         subject_tests=subject_tests,
                         student_col=student_col,
                         roll_col=roll_col,
                         enrollment_col=enrollment_col,
                         strongest=strongest, 
                         weakest=weakest, 
                         most_improved=most_improved, 
                         least_improved=least_improved, 
                         highest_per_test=highest_per_test, 
                         overall_topper=overall_topper,
                         top_students=top_students.to_dict(orient='records'),
                         subject_analysis=subject_analysis,
                         most_consistent=most_consistent,
                         most_deteriorated=most_deteriorated,
                         subject_statistics=subject_statistics,
                         division_statistics=division_statistics)

def generate_student_graphs(student_data):
    """Generate graphs for individual student performance"""
    
    student_name = student_data['student_info'][student_col].replace(' ', '_')
    graphs_dir = "static/student_graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    
    plt.style.use('ggplot')  
    
    fig, ax = plt.subplots(figsize=(10, 5))
    subjects = list(student_data['subject_performance'].keys())
    averages = [perf['average'] for perf in student_data['subject_performance'].values()]
    subject_avgs = [df[subject_tests[subject]].mean().mean() for subject in subjects]
    
    x = np.arange(len(subjects))
    width = 0.35
    
    ax.bar(x - width/2, averages, width, label='Student Average', color='#2ecc71')
    ax.bar(x + width/2, subject_avgs, width, label='Class Average', color='#3498db')
    
    ax.set_ylabel('Average Marks')
    ax.set_title('Subject-wise Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()
    
    for i, v in enumerate(averages):
        ax.text(i - width/2, v, f'{v:.1f}', ha='center', va='bottom')
    for i, v in enumerate(subject_avgs):
        ax.text(i + width/2, v, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    performance_graph = f"{graphs_dir}/{student_name}_performance.png"
    plt.savefig(performance_graph, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    
    for (subject, perf), color in zip(student_data['subject_performance'].items(), colors):
        marks = list(perf['test_marks'].values())
        ax.plot([1, 2, 3, 4], marks, marker='o', label=subject, color=color, linewidth=2)
        
        for i, mark in enumerate(marks, 1):
            ax.annotate(f'{mark}', (i, mark), textcoords="offset points", 
                       xytext=(0,10), ha='center')
    
    ax.set_xlabel('Test Number')
    ax.set_ylabel('Marks')
    ax.set_title('Progress Across Tests')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Test 1', 'Test 2', 'Test 3', 'Test 4'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    progress_graph = f"{graphs_dir}/{student_name}_progress.png"
    plt.savefig(progress_graph, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'performance_graph': performance_graph.replace('static/', ''),
        'progress_graph': progress_graph.replace('static/', '')
    }

@app.route('/student', methods=['GET'])
def student():
    enrollment_no = request.args.get('enrollment')
    if not enrollment_no:
        return "Please provide enrollment number"
    
    enrollment_no = str(enrollment_no).strip()
    print(f"\nSearching for enrollment number: {enrollment_no}")
    
    student_data = calculate_student_performance(enrollment_no)
    if not student_data:
        return f"""
        <h3>Student not found with enrollment number: {enrollment_no}</h3>
        <p>Please check the enrollment number and try again.</p>
        <p><a href="/">Back to Dashboard</a></p>
        """
    
    analysis = analyze_student_performance(student_data)
    total_students = len(df)  
    
    graphs = generate_student_graphs(student_data)
    
    return render_template('student.html', 
                         data=student_data,
                         analysis=analysis,
                         total_students=total_students,
                         student_col=student_col,
                         enrollment_col=enrollment_col,
                         roll_col=roll_col,
                         graphs=graphs)

@app.route('/download/<enrollment_no>')
def download(enrollment_no):
    enrollment_no = enrollment_no.strip()
    
    student_data = calculate_student_performance(enrollment_no)
    if not student_data:
        return "Student not found"
    
    pdf_path = generate_marksheet(student_data)
    
    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{student_data['student_info'][student_col].replace(' ', '_')}_marksheet.pdf"
    )

@app.route('/download_subject_result/<subject>/<status>')
def download_subject_result(subject, status):
    passing_marks = 35
    subject_tests_list = subject_tests[subject]
    
    df['Subject_Total'] = df[subject_tests_list].fillna(0).sum(axis=1)
    
    passing_students = df[df['Subject_Total'] >= passing_marks].sort_values('Subject_Total', ascending=False)
    failing_students = df[(df['Subject_Total'] < passing_marks) & (df['Subject_Total'] > 0)].sort_values('Subject_Total', ascending=False)
    
    columns_to_export = [student_col, roll_col, enrollment_col] + list(subject_tests_list) + ['Subject_Total']
    
    result_df = (passing_students if status == 'pass' else failing_students)[columns_to_export]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'{subject}_{status}_list.xlsx'
    )

@app.route('/download_division_list/<division>')
def download_division_list(division):
    sorted_df = df.sort_values('Total Marks', ascending=False)
    students_per_division = 30
    
    if division == 'A':
        division_df = sorted_df.iloc[0:students_per_division]
        start_roll = 1
    elif division == 'B':
        division_df = sorted_df.iloc[students_per_division:2*students_per_division]
        start_roll = 31
    elif division == 'C':
        division_df = sorted_df.iloc[2*students_per_division:3*students_per_division]
        start_roll = 61
    else:  
        division_df = sorted_df.iloc[3*students_per_division:]
        start_roll = 91
    
    export_df = division_df.copy()
    export_df['New Roll No'] = range(start_roll, start_roll + len(export_df))
    
    columns_to_export = ['New Roll No', student_col, enrollment_col]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df[columns_to_export].to_excel(writer, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'Division_{division}_list.xlsx'
    )

if not os.path.exists("static"):
    os.makedirs("static")

def generate_graphs():
    avg_marks = {subject: df[tests].mean().mean() for subject, tests in subject_tests.items()}
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(avg_marks.keys()), y=list(avg_marks.values()), palette="coolwarm")
    plt.xlabel("Subjects")
    plt.ylabel("Average Marks")
    plt.title("Subject-wise Average Marks")
    plt.savefig("static/subject_avg.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for subject, tests in subject_tests.items():
        sns.lineplot(x=[1, 2, 3, 4], y=df[tests].mean(), label=subject, marker="o")
    plt.xticks([1, 2, 3, 4], ["Test 1", "Test 2", "Test 3", "Test 4"])
    plt.xlabel("Test Number")
    plt.ylabel("Average Marks")
    plt.title("Test-wise Performance Trend")
    plt.legend()
    plt.savefig("static/test_trend.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["Total Marks"], bins=10, kde=True, color="purple")
    plt.xlabel("Total Marks")
    plt.ylabel("Number of Students")
    plt.title("Distribution of Total Marks")
    plt.savefig("static/marks_distribution.png")
    plt.close()

    for subject in subjects:
        plt.figure(figsize=(12, 3))
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        fig.suptitle(f'{subject} - Test-wise Marks Distribution')
        
        for i, test in enumerate(subject_tests[subject]):
            test_marks = df[test].fillna(0)  
            sns.histplot(data=test_marks[test_marks > 0], bins=10, kde=True, ax=axes[i])  
            axes[i].set_title(f'Test {i+1}')
            axes[i].set_xlabel('Marks')
            axes[i].set_ylabel('Number of Students')
            axes[i].set_xlim(0, 25)  
        
        plt.tight_layout()
        plt.savefig(f"static/{subject.lower()}_test_analysis.png")
    plt.close()

generate_graphs()


if __name__ == '__main__':
    app.run(debug=True)

