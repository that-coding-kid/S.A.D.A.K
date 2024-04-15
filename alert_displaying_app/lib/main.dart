import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Simple App',
      initialRoute: '/',
      routes: {
        '/': (context) => FirstScreen(),
        '/second': (context) => SecondScreen(),
        '/third': (context) => ThirdScreen(),
      },
    );
  }
}

class FirstScreen extends StatefulWidget {
  @override
  _FirstScreenState createState() => _FirstScreenState();
}

class _FirstScreenState extends State<FirstScreen> {
  bool encroachmentFound = false; // Variable to track if encroachment is found

  @override
  void initState() {
    super.initState();
    // Call your ML model to check for encroachment here
    // For example:
    // checkForEncroachment();
  }

  void checkForEncroachment() {
    // Assume this is where you call your ML model to check for encroachment
    // Replace this with your actual code to call the ML model
    // For demonstration purpose, I'm setting it to true after 5 seconds
    Future.delayed(Duration(seconds: 5), () {
      setState(() {
        encroachmentFound = true;
        if (encroachmentFound) {
          Navigator.pushNamed(context, '/second');
        }
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(''), 
        leading: Image.asset(
          'assets/Screenshot_2024-04-13_145725-removebg.png', 
          width: 1000, 
          height: 1000, 
        ),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'NOTHING TO SHOW HERE (NO BOTTLENECK FOUND!)',
              style: TextStyle(fontSize: 24.0),
            ),
            ElevatedButton(
              onPressed: () {
                // Do nothing on button press, encroachment check is automatic
              },
              child: Text('Checking for Encroachment...'),
            ),
          ],
        ),
      ),
    );
  }
}

class SecondScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Second Screen'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'ENCROACHMENT FOUND!',
              style: TextStyle(fontSize: 24.0),
            ),
            Text(
              'Location: ',
              style: TextStyle(fontSize: 16.0),
            ),
            Text(
              'Image: ',
              style: TextStyle(fontSize: 16.0),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/third');
              },
              child: Text('Send a tow truck operator'),
            ),
          ],
        ),
      ),
    );
  }
}

class ThirdScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Third Screen'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Required authority is sent to the given location!',
              style: TextStyle(fontSize: 24.0),
            ),
          ],
        ),
      ),
    );
  }
}
