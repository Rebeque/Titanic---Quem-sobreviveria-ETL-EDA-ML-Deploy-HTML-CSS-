import flask
import pickle
import pandas as pd

# Lê modelo usando pickle
with open(f'model/modelo_final_v4.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        Pclass = flask.request.form['Pclass']
        Sex = flask.request.form['Sex']
        Embarked = flask.request.form['Embarked']

        input_variables = pd.DataFrame([[Pclass, Sex, Embarked]], columns=['Pclass', 'Sex', 'Embarked']).astype('int64')
        prediction = model.predict(input_variables)[0]

    if prediction == 0:
        return flask.render_template('main.html', original_input={'Classe de Embarque':Pclass, 'Sexo':Sex, 'Estação de Embarque':Embarked}, result=('Morreu'))

    if prediction == 1:
        return flask.render_template('main.html', original_input={'Classe de Embarque':Pclass, 'Sexo':Sex, 'Estação de Embarque':Embarked}, result=('Sobreviveu'))


if __name__ == '__main__':
    app.run()
