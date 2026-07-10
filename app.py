"""
Gradio app for the Craigslist car price prediction model.

Before running this:
1. In your notebook, save the preprocessor alongside the model:
       joblib.dump(processor, "car_price_processor.pkl")
2. Put both "best_random_forest.pkl" and "car_price_processor.pkl"
   in the same folder as this script.
3. Run:  python app.py
"""

import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = "best_random_forest.pkl"
PROCESSOR_PATH = "car_price_processor.pkl"

model = joblib.load(MODEL_PATH)
processor = joblib.load(PROCESSOR_PATH)

# Category options taken from your df[col].value_counts() output in the notebook.
# region/model have thousands of unique values, so they're free-text inputs;
# the TargetEncoder falls back to the global mean for unseen values, so any
# text is safe to enter.
MANUFACTURERS = [
    "ford", "chevrolet", "toyota", "honda", "nissan", "jeep", "ram", "gmc",
    "bmw", "dodge", "mercedes-benz", "hyundai", "subaru", "volkswagen",
    "lexus", "kia", "audi", "cadillac", "acura", "chrysler", "buick",
    "mazda", "infiniti", "lincoln", "volvo", "mitsubishi", "mini", "rover",
    "jaguar", "pontiac", "porsche", "saturn", "mercury", "tesla",
    "alfa-romeo", "fiat", "harley-davidson", "ferrari", "aston-martin",
    "land rover", "morgan",
]
CONDITIONS = ["Unknown", "good", "excellent", "like new", "fair", "new", "salvage"]
CYLINDERS = [
    "Unknown", "6 cylinders", "4 cylinders", "8 cylinders", "5 cylinders",
    "10 cylinders", "other", "3 cylinders", "12 cylinders",
]
FUELS = ["gas", "other", "diesel", "hybrid", "electric"]
TITLE_STATUSES = ["clean", "rebuilt", "salvage", "lien", "missing", "parts only"]
TRANSMISSIONS = ["automatic", "other", "manual"]
DRIVES = ["4wd", "Unknown", "fwd", "rwd"]
TYPES = [
    "sedan", "Unknown", "SUV", "pickup", "truck", "other", "coupe",
    "hatchback", "wagon", "van", "convertible", "mini-van", "offroad", "bus",
]
PAINT_COLORS = [
    "Unknown", "white", "black", "silver", "blue", "red", "grey", "green",
    "brown", "custom", "orange", "yellow", "purple",
]
STATES = [
    "ca", "fl", "tx", "ny", "oh", "mi", "or", "pa", "nc", "wa", "wi", "tn",
    "co", "il", "nj", "va", "ia", "id", "az", "ma", "mn", "ga", "ks", "mt",
    "sc", "ok", "in", "ct", "al", "md", "ky", "mo", "nm", "ak", "ar", "nv",
    "nh", "dc", "la", "hi", "me", "vt", "ri", "sd", "ut", "wv", "ne", "ms",
    "de", "wy", "nd",
]

# Most common posting_date in the training data — used as a stand-in default,
# since TargetEncoder maps unseen dates to the global mean anyway.
DEFAULT_POSTING_DATE = "2021-04-23 22:13:05-04:00"

# Column order must match x_train = df.drop(['price'], axis=1) exactly.
COLUMN_ORDER = [
    "region", "year", "manufacturer", "model", "condition", "cylinders",
    "fuel", "odometer", "title_status", "transmission", "drive", "type",
    "paint_color", "state", "posting_date",
]


def predict_price(
    region, year, manufacturer, model_name, condition, cylinders, fuel,
    odometer, title_status, transmission, drive, vehicle_type, paint_color,
    state,
):
    row = {
        "region": region.strip() if region else "unknown",
        "year": float(year),
        "manufacturer": manufacturer,
        "model": model_name.strip() if model_name else "unknown",
        "condition": condition,
        "cylinders": cylinders,
        "fuel": fuel,
        "odometer": float(odometer),
        "title_status": title_status,
        "transmission": transmission,
        "drive": drive,
        "type": vehicle_type,
        "paint_color": paint_color,
        "state": state,
        "posting_date": pd.to_datetime(DEFAULT_POSTING_DATE),
    }
    input_df = pd.DataFrame([row])[COLUMN_ORDER]

    x_transformed = processor.transform(input_df)
    prediction = model.predict(x_transformed)[0]

    return f"${prediction:,.2f}"


with gr.Blocks(title="Used Car Price Predictor") as demo:
    gr.Markdown("# 🚗 Used Car Price Predictor")
    gr.Markdown(
        "Enter the details of a used car to estimate its market price, "
        "based on a Random Forest model trained on Craigslist listings."
    )

    with gr.Row():
        with gr.Column():
            region = gr.Textbox(label="Region", placeholder="e.g. columbus")
            manufacturer = gr.Dropdown(MANUFACTURERS, label="Manufacturer", value="ford")
            model_name = gr.Textbox(label="Model", placeholder="e.g. f-150")
            year = gr.Slider(1997, 2021, value=2015, step=1, label="Year")
            odometer = gr.Number(label="Odometer (miles)", value=50000)
            condition = gr.Dropdown(CONDITIONS, label="Condition", value="good")
            cylinders = gr.Dropdown(CYLINDERS, label="Cylinders", value="6 cylinders")
            fuel = gr.Dropdown(FUELS, label="Fuel Type", value="gas")

        with gr.Column():
            title_status = gr.Dropdown(TITLE_STATUSES, label="Title Status", value="clean")
            transmission = gr.Dropdown(TRANSMISSIONS, label="Transmission", value="automatic")
            drive = gr.Dropdown(DRIVES, label="Drive", value="4wd")
            vehicle_type = gr.Dropdown(TYPES, label="Vehicle Type", value="sedan")
            paint_color = gr.Dropdown(PAINT_COLORS, label="Paint Color", value="white")
            state = gr.Dropdown(STATES, label="State", value="ca")

    predict_btn = gr.Button("Predict Price", variant="primary")
    output = gr.Textbox(label="Estimated Price")

    predict_btn.click(
        fn=predict_price,
        inputs=[
            region, year, manufacturer, model_name, condition, cylinders,
            fuel, odometer, title_status, transmission, drive, vehicle_type,
            paint_color, state,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
