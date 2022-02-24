from typing import Literal
from pydantic import BaseModel


class SQuADInput(BaseModel):
    context: str = """Oxygen is a chemical element with symbol O and atomic number 8. It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements. By mass, oxygen is the third-most abundant element in the universe, after hydrogen and helium.
    At standard temperature and pressure, two atoms of the element bind to form dioxygen, a colorless and odorless diatomic gas with the formula O
    2. Diatomic oxygen gas constitutes 20.8% of the Earth's atmosphere.
    However, monitoring of atmospheric oxygen levels show a global downward trend, because of fossil-fuel burning.
    Oxygen is the most abundant element by mass in the Earth's crust as part of oxide compounds such as silicon dioxide, making up almost half of the crust's mass."""
    question: str = "The atomic number of the periodic table for oxygen?"
    model_to_use: Literal['without_impossible', 'with_impossible']


class SQuADOutput(BaseModel):
    answer: str = '8'
