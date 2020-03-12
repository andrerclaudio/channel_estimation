import logging

# Log parameters adjustment call
# _ Hour (24 hours format)
# _ Minutes
# _ Seconds
# _ Month-Day
# _ Level to print and above
# _ Message to show

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] - %(levelname)s:  %(message)s',
                    datefmt='%H:%M:%S %m-%d-%y')


def application():
    """" All application has its initialization from here """
    logging.info('Main application is running!')
