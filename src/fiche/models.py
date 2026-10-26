from pydantic import BaseModel
from typing import Optional, List


class Fiche(BaseModel):
    fiche_id: str

    seviye: str
    kurum: str
    fakulte: Optional[str]
    bolum: Optional[str]

    belge_turu: str
    yonerge_id: str
    yonerge_adi: str

    bolum_no: Optional[int]
    bolum_basligi: Optional[str]

    madde_no: int
    madde_basligi: Optional[str]
    fikra_no: Optional[int]
    bent: Optional[str]

    metin: str
    embedding_text: str

    tags: List[str] = []
    version: int = 1