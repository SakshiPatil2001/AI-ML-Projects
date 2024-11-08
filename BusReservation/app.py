# Importing Required Modules
# Flask: A lightweight web framework for Python.
# render_template: Renders HTML templates.
# request: Handles incoming HTTP requests.
# redirect: Redirects the user to a different URL after an action.
# get_db_connection: Custom function to get a connection to the MySQL database.
from flask import Flask, render_template, request, redirect
from db_config import get_db_connection

# Initialize the Flask app
# app: The Flask app instance that will handle incoming requests and send responses.
app = Flask(__name__)

# Route for the Home Page
# When users visit the root URL '/', this function will be triggered and will return the home page template.
@app.route('/')
def home():
    # Renders 'home.html' which is the homepage for the application.
    return render_template('home.html')

# Route for Booking Tickets
# This route handles both GET and POST requests. When accessed via GET, it shows the booking form, and when submitted via POST, it processes the booking.
@app.route('/book', methods=['GET', 'POST'])
def book_ticket():
    if request.method == 'POST':
        # Extract data from the booking form
        passenger_name = request.form['passenger_name']
        bus_number = request.form['bus_number']
        seat_number = request.form['seat_number']
        date_of_travel = request.form['date_of_travel']

        # Connect to the MySQL database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert the booking data into the 'bookings' table
        cursor.execute('INSERT INTO bookings (passenger_name, bus_number, seat_number, date_of_travel) VALUES (%s, %s, %s, %s)',
               (passenger_name, bus_number, seat_number, date_of_travel))

        # Commit the changes and close the database connection
        conn.commit()
        cursor.close()
        conn.close()

        # Redirect the user to the confirmation page after booking
        return redirect('/confirmation')

    # If the request method is GET, display the booking form
    return render_template('book.html')

# Route for the Confirmation Page
# This function displays a confirmation message after the booking is successfully processed.
@app.route('/confirmation')
def confirmation():
    # Renders the 'confirmation.html' page that shows booking success information.
    return render_template('confirmation.html')

# Run the application
# This line runs the Flask application with debugging enabled. Debug mode auto-reloads the app during development.
if __name__ == '__main__':
    app.run(debug=True)

