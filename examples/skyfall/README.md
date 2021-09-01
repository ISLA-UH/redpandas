### Skyfall

Skyfall is an event where a smartphone fell from a high altitude and landed on the ground.

Using the data from this event, RedPandas is able to produce several products.

To begin, download the skyfall data from [the Redvox website](http://redvox.io/@/3f3f).

Next, open `skyfall_config_file.py` and update the value of `INPUT_DIR` on line 5 to match the directory 
where you downloaded the skyfall data (the directory will have a folder named "api900"). An example is below:
```python
INPUT_DIR = "/path_to/your/downloaded_data/"
```

Now, run the `skyfall_intro.py` file.  This will create simple products for you to view.

For more RedPandas products, run the `run_all.py` file.

You may view and run the specific example functions in the `lib/` directory.