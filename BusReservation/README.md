# Bus Ticket Reservation Web Application

This is a Flask-based web application for booking bus tickets. Users can input their details such as passenger name, bus number, seat number, and date of travel to reserve a seat. The booking information is stored in a MySQL database.

## Features

- Simple form for booking tickets.
- Booking information is saved into a MySQL database.
- A confirmation page is displayed after a successful booking.

## Technologies Used

- **Flask** (Python)
- **HTML** (For the front-end templates)
- **MySQL** (For storing booking information)


## Installation and Setup Instructions

1. Clone the Repository
To get started with the project, clone the repository to your local machine:
```bash
git clone https://github.com/SakshiPatil2001/Sakshi-Patil-AIML-Projects.git
cd "zomato price prediction"

2. Install Dependencies 
pip install flask

3. Set Up the Database
CREATE DATABASE bus_reservation;
USE bus_reservation;
CREATE TABLE bookings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    passenger_name VARCHAR(100),
    bus_number VARCHAR(10),
    seat_number VARCHAR(10),
    date_of_travel DATE
);

4. Running the Application
   python app.py

5. Usage
Go to the homepage (http://127.0.0.1:5000/).
Click on the "Book a Ticket" button and fill in the booking form.
Submit the form, and you will be redirected to the confirmation page.


