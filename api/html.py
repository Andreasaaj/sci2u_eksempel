
TOP_TEMPLATE = '''
<html>
<head>
<title>Preview!</title>
<style type="text/css">
img.question { border: 4px solid black; }
img.answer { width: 150px; border: 4px solid white; border: 4px solid red; }
img.answer.correct { border: 4px solid green; }
div.answers { float: right; max-width: 350px; }
</style>
</head>
<body>

<h1>Preview</h1>

%s

</body>
</html>'''


QUESTION_TEMPLATE = '''
<div>
    <img class="question" src="%s" />
    <div class="answers">
        %s
    </div>
</div>

<hr>
'''


ANSWER_TEMPLATE = '''
<img src="%s" class="answer %s" />
'''


def make_html_preview(help_sheets, question_answer_pairs, filepath):
    questions_html_parts = []

    for help_sheet in help_sheets:
        answers_html = ''
        question_html = QUESTION_TEMPLATE % (help_sheet._filename, answers_html)
        questions_html_parts.append(question_html)
        
    for question, answers in question_answer_pairs:
        answers_html_parts = []
        
        for answer in answers:
            if question.is_correct_answer(answer):
                cls = 'correct'
            else:
                cls = ''
            
            answers_html_parts.append(ANSWER_TEMPLATE % (answer._filename, cls))
        
        answers_html = ''.join(answers_html_parts)
        question_html = QUESTION_TEMPLATE % (question._filename, answers_html)
        questions_html_parts.append(question_html)
    
    questions_html = ''.join(questions_html_parts)
    html = TOP_TEMPLATE % questions_html
    
    with open(filepath, 'w') as htmlfile:
        htmlfile.write(html)

