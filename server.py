from flask import *
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)


password = "a1"

predicts = None
names = None

student_name = 'Ерохина'
student_score = "Биоинформатика"

UPLOAD_FOLDER = '/Users/VladiYar/PycharmProjects/Neural-get/static'
ALLOWED_EXTENSIONS = set(['csv'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/")
def start():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ""
    if request.method == 'POST':
        if request.form['selector'] == 'teacher':
            if request.form["login"] == password:
                session['logged_in'] = True
                return redirect(url_for('get_data'))
            else:
                error = " Неверный логин/пароль"
        elif request.form['selector'] == 'student':
            return redirect(url_for('student', id=request.form['id']))
    return render_template('login.html', error=error)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data=[]
    for i in len(predicts):
        data.append([predicts[i] + names[i]])
    return render_template('predict.html', data=data)


@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    if request.method == "POST":
        data = request.files['table']
        if data and allowed_file(data.filename):
            filename = secure_filename(data.filename)
            data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            data_csv = pd.read_csv(app.config['UPLOAD_FOLDER'] + '/' + filename)

            names = np.array(data_csv['Ваше ФИО'])

            pedicts = pred(np.array(data_csv['Возраст (1-18)']), np.array(data_csv['Какие проекты Вы реализовали?']),
                           np.array(data_csv['В каких олимпиадах Вы участвовали?']),
                           np.array(data_csv['Какие языки программирования Вы знаете?']))

            return redirect(url_for('predict'))
        else:
            return render_template("get_data.html", error='Неверный формат данных')
    return render_template("get_data.html", error="")


@app.route('/student/<id>', methods=['GET', 'POST'])
def student(id):
    path = {
        "Биоинформатика": url_for('static', filename='bio.png'),
        "Робототехника": url_for('static', filename='robots.png'),
        "Анализ данных": url_for('static', filename='ad.png'),
        "Прикладное программирование": url_for('static', filename='pp1.png')
    }
    return render_template('student.html', name=student_name, score=student_score, path=path[student_score])


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('start'))


app.secret_key = os.urandom(24)


if __name__ == '__main__':
    app.run(debug=True)
