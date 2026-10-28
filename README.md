# Van Gogh AI Studio

A Streamlit app for turning uploaded photographs into Van Gogh-inspired
paintings.

## Run the app

Install the packages listed in `requirements.txt`, then start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app can run either the fine-tuned generator or the original pretrained
generator on the uploaded photograph. The two checkpoints use the same
CycleGAN generator architecture but contain different weights. Output
dimensions match the uploaded photograph; resizing does not add image detail.

The app uses `assets/starry-night-background.png` as its page background.

## Model files

Place the checkpoints in these exact locations:

```text
VanGogh_Streamlit/
├── app.py
├── model.py
└── models/
    ├── fine_tuned/
    │   └── vangogh_generator.pth
    └── original_pretrained/
        └── verified_vangogh_generator.pth
```

The fine-tuned button loads only `models/fine_tuned/vangogh_generator.pth`.
The original pretrained button loads only
`models/original_pretrained/verified_vangogh_generator.pth`. Generated PNGs
are saved in `outputs/` and can also be downloaded from the comparison view.