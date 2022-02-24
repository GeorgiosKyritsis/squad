import uvicorn
from fastapi import FastAPI

from squad.model import SQuADModel

from webapp.models import SQuADInput, SQuADOutput


models_dict = {}
model_without_impossible = SQuADModel.load_model('models/without_impossible/model.ckpt')
models_dict['without_impossible'] = model_without_impossible

model_with_impossible = SQuADModel.load_model('models/with_impossible/model.ckpt')
models_dict['with_impossible'] = model_with_impossible


app = FastAPI()


@app.post('/get-answer', response_model=SQuADOutput)
def get_answer(parameters: SQuADInput):
    answer = models_dict[parameters.model_to_use].generate_answers(
        [{"context": parameters.context, "question": parameters.question}],
        batch_size=2
    )

    return {'answer': str(answer[0])}


if __name__ == '__main__':
    uvicorn.run("webapp.back:app", port=1111, host='0.0.0.0')
