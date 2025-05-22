from typing import Optional

import pandera as pa
from pandera.typing import Series

class CoefSchema(pa.DataFrameSchema):
    n: Series[int]
    A: Optional[Series[float]]
    A2: Optional[Series[float]]
    A3: Optional[Series[float]]
    A4: Optional[Series[float]]
    B3: Optional[Series[float]]
    B4: Optional[Series[float]]
    B5: Optional[Series[float]]
    B6: Optional[Series[float]]
    D1: Optional[Series[float]]
    D2: Optional[Series[float]]
    D3: Optional[Series[float]]
    D4: Optional[Series[float]]
    E2: Optional[Series[float]]
    c4: Optional[Series[float]]
    d2: Optional[Series[float]]

class XRDataSheetSchema(pa.DataFrameSchema):
    X1: Series[int]
    X2: Series[int]
    X3: Optional[Series[int]]
    X4: Optional[Series[int]]
    X5: Optional[Series[int]]
    X6: Optional[Series[int]]
    X7: Optional[Series[int]]
    X8: Optional[Series[int]]
    X9: Optional[Series[int]]
    X10: Optional[Series[int]]

class XRsDataSheetSchema(pa.DataFrameSchema):
    X: Series[float]

class npDataSheetSchema(pa.DataFrameSchema):
    np: Series[int]

class pDataSheetSchema(pa.DataFrameSchema):
    n: Series[int]
    np: Series[int]

class cDataSheetSchema(pa.DataFrameSchema):
    c: Series[int]

class uDataSheetSchema(pa.DataFrameSchema):
    n: Series[int]
    c: Series[int]
