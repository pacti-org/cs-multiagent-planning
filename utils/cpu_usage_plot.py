import psutil
import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from IPython.display import display as ipyd, clear_output, Image as IPythonImage
import ipywidgets as widgets
import threading
from contextlib import contextmanager
from io import BytesIO

# Function to get the current CPU usage percentage
def get_cpu_usage():
    return psutil.cpu_percent(interval=0.1)

@contextmanager
def cpu_usage_plot(max_data_points: int = 200):
    backend = matplotlib.get_backend()
    matplotlib.use('agg')

    fig, ax = plt.subplots(figsize=(8, 4))
    # the upper limit is 101 so that we can see data points with y=100
    ax.set_ylim(0, 101) 
    ax.set_xlabel("Time")
    ax.set_ylabel("CPU Usage (%)")

    cpu_usage_data = []
    time_data = []
    start_time = time.time()
    line, = ax.plot(cpu_usage_data)
    fill = ax.fill_between(range(len(cpu_usage_data)), cpu_usage_data, 0, alpha=0.3)

    img = widgets.Image()
    display_handle = ipyd(img, display_id=True)

    def update_plot():
        nonlocal cpu_usage_data, time_data
        cpu_usage = get_cpu_usage()
        cpu_usage_data.append(cpu_usage)
        current_time = time.time() - start_time
        time_data.append(current_time)

        if len(cpu_usage_data) > max_data_points:
            cpu_usage_data.pop(0)
            time_data.pop(0)

        ax.clear()
        # bump the limit by one to see data with y=100
        ax.set_ylim(0, 101)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("CPU Usage (%)")
        
        line, = ax.plot(time_data, cpu_usage_data)
        ax.fill_between(time_data, cpu_usage_data, 0, alpha=0.3)

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img.value = buf.read()

        display_handle.update(img)

        ax.relim()
        ax.autoscale_view()


    def stop_condition():
        nonlocal stop
        return stop

    def display_cpu_usage_plot_widget(stop_condition):
        nonlocal stop
        try:
            while not stop_condition():
                update_plot()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    stop = False
    cpu_usage_plot_thread = threading.Thread(target=display_cpu_usage_plot_widget, args=(stop_condition,), daemon=True)
    cpu_usage_plot_thread.start()

    output_widget = widgets.Output()
    ipyd(output_widget)

    stored_output = widgets.Output()
    
    # Create a VBox to stack output_widget and stored_output vertically
    display_widget = widgets.VBox([output_widget, stored_output])
    ipyd(display_widget)

    try:
        with output_widget:
            with stored_output:
                yield
    finally:
        stop = True
        cpu_usage_plot_thread.join()
        clear_output(wait=True)

    # Display the final plot as part of the notebook

    final_buf = BytesIO()
    fig.savefig(final_buf, format='png')
    final_buf.seek(0)
    plt.close(fig) 
    
    matplotlib.use(backend)
    ipyd(IPythonImage(data=final_buf.getvalue()))

    ipyd(stored_output)
