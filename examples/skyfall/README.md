### Examples: Skyfall

The Skyfall data is a high-quality dataset that showcases the RedPandas library for processing smartphone data collected 
with the [RedVox Infrasound Recorder app](https://www.redvoxsound.com).

In essence, Skyfall is an event where a smartphone fell from a high altitude and landed on the ground. A balloon hoisted a commercial, off-the-shelf, smartphone to a height of 36 km (around 119,000 feet) and purposely burst
to let the smartphone freefall (hence the name _Skyfall_). As the smartphone fell back to Earth, it recorded its 30 minute 
descent using the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. 

Using the data from this event, RedPandas is able to produce several products.

#### Downloading the RedVox Skyfall data
To begin, download the Skyfall data from [the Redvox website](http://redvox.io/@/3f3f). Under "RedVox Report", you will 
find "Download Report Data" towards the end.

#### Running the Skyfall example
Next, open `skyfall_config_file.py` and update the value of `SKYFALL_DIR` on line 5 to match the directory 
where you downloaded the Skyfall data (the directory will have a folder named "api900").

An example path for Linux/Mac:
```python
SKYFALL_DIR = "/path_to/your/downloaded_data"
```

An example path for Windows:
```python
SKYFALL_DIR = "\\path_to\\your\\downloaded_data"
```

Now, run the `skyfall_intro.py` file.  This will create some products for you to view.

For more RedPandas products, run the `run_all.py` file in the `lib/` directory.

You may view and run the specific example functions (`skyfall_*.py`) in the `lib/` directory.