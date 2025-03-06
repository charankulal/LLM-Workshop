# Day 1
## main.dart

```dart
import 'package:flutter/material.dart';
import 'pages/home_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'wally',
      theme: ThemeData(
        brightness: Brightness.dark,
      ),
      home:HomePage(),
    );
  }
}
```


### Explanation of Components and Widgets

#### 1. **`MaterialApp` (Root Widget)**
   - The main widget that initializes the app with Material Design.
   - Options used:
     - `title`: Sets the app title.
     - `theme`: Defines the app's theme, here set to `Brightness.dark`.
     - `home`: Specifies the starting screen (`HomePage`).

#### 2. **`MyApp` (Stateless Widget)**
   - A stateless widget that returns `MaterialApp`.
   - Uses a constant constructor (`const MyApp({super.key})`) for performance efficiency.

#### 3. **`HomePage` (Custom Widget)**
   - Imported from `'pages/home_page.dart'`.
   - Used as the `home` screen of `MaterialApp`.

#### 4. **`runApp(MyApp())` (Entry Point)**
   - Calls the `MyApp` widget and starts the Flutter application.


## pages/home_page.dart

```dart
import 'package:flutter/material.dart';
import '../wallpapers.dart';
import '../widgets/scrollable_wallpaper_widget.dart';

class HomePage extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    return _HomePageState();
  }
}

class _HomePageState extends State<HomePage> {
  var _deviceHeight;
  var _deviceWidth;
  var _selectedWallPaper;

  @override
  void initState() {
    _selectedWallPaper = 0;
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    _deviceHeight = MediaQuery.of(context).size.height;
    _deviceWidth = MediaQuery.of(context).size.width;
    return Scaffold(
      body: Stack(
        children: <Widget>[
          _featuredWallPapers(),
          _gradientBoxWidget(),
          _topLayerWidget(),
        ],
      ),
    );
  }

  Widget _featuredWallPapers() {
    return SizedBox(
      height: _deviceHeight * 0.5,
      width: _deviceWidth,
      child: PageView(
        onPageChanged: (_index) {
          setState(() {
            _selectedWallPaper = _index;
          });
        },
        scrollDirection: Axis.horizontal,
        children:
            featuredWallPapers.map((_wallpaper) {
              return Container(
                decoration: BoxDecoration(
                  image: DecorationImage(
                    fit: BoxFit.cover,
                    image: NetworkImage(_wallpaper.coverImage.url),
                  ),
                ),
              );
            }).toList(),
      ),
    );
  }

  Widget _gradientBoxWidget() {
    return Align(
      alignment: Alignment.bottomCenter,
      child: Container(
        height: _deviceHeight * 0.75,
        width: _deviceWidth,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Color.fromRGBO(35, 45, 60, 1.0), Colors.transparent],
            stops: [0.65, 1.0],
            begin: Alignment.bottomCenter,
            end: Alignment.topCenter,
          ),
        ),
      ),
    );
  }

  Widget _topLayerWidget() {
    return Padding(
      padding: EdgeInsets.symmetric(
        horizontal: _deviceWidth * 0.05,
        vertical: _deviceWidth * 0.05,
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        mainAxisSize: MainAxisSize.max,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          _topBarWidget(),
          SizedBox(height: _deviceHeight * 0.13),
          _featuredWallPapersInfoWidget(),
          Padding(
            padding: EdgeInsets.symmetric(vertical: _deviceHeight * 0.001),
            child: ScrollableWallPaperWidget(
              _deviceHeight * 0.22,
              _deviceWidth,
              true,
              wallpapers,
            ),
          ),
          _featuredWallPaperBannerWidget(),
      Padding(
        padding: EdgeInsets.symmetric(vertical: _deviceHeight * 0.001),
        child: ScrollableWallPaperWidget(
          _deviceHeight * 0.21,
          _deviceWidth,
          false,
          wallpapers2,
        ),
      )
        ],
      ),
    );
  }

  Widget _topBarWidget() {
    return SizedBox(
      height: _deviceHeight * 0.13,
      width: _deviceWidth,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        mainAxisSize: MainAxisSize.max,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Icon(Icons.menu, color: Colors.white, size: 30),
          Row(
            children: <Widget>[
              Icon(Icons.search, color: Colors.white, size: 30),
              SizedBox(width: _deviceWidth * 0.03),
              Icon(Icons.notifications, color: Colors.white, size: 30),
            ],
          ),
        ],
      ),
    );
  }

  Widget _featuredWallPapersInfoWidget() {
    double _circleRadius = _deviceHeight * 0.0022;
    return SizedBox(
      height: _deviceHeight * 0.12,
      width: _deviceWidth,
      child: Column(
        mainAxisSize: MainAxisSize.max,
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            featuredWallPapers[_selectedWallPaper].title,
            maxLines: 2,
            style: TextStyle(
              color: Colors.white,
              fontSize: _deviceHeight * 0.04,
            ),
          ),
          SizedBox(height: _deviceHeight * 0.0001),
          Row(
            mainAxisAlignment: MainAxisAlignment.start,
            mainAxisSize: MainAxisSize.max,
            crossAxisAlignment: CrossAxisAlignment.center,
            children:
                featuredWallPapers.map((_wallpaper) {
                  bool _isActive =
                      _wallpaper.title ==
                      featuredWallPapers[_selectedWallPaper].title;
                  return Container(
                    margin: EdgeInsets.only(right: _deviceWidth * 0.015),
                    height: _circleRadius * 2,
                    width: _circleRadius * 2,
                    decoration: BoxDecoration(
                      color: _isActive ? Colors.green : Colors.grey,
                      borderRadius: BorderRadius.circular(100),
                    ),
                  );
                }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _featuredWallPaperBannerWidget() {
    return Container(
      height: _deviceHeight * 0.13,
      width: _deviceWidth,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(5),
        image: DecorationImage(
          fit: BoxFit.cover,
          image: NetworkImage(featuredWallPapers[2].coverImage.url),
        ),
      ),
    );
  }
}
```


### Explanation of Components and Widgets

#### 1. **`HomePage` (Stateful Widget)**
   - A stateful widget that manages the state of the home screen.
   - Uses `_HomePageState` to handle UI updates dynamically.

#### 2. **`_deviceHeight` and `_deviceWidth`**
   - Stores the device's height and width using `MediaQuery`.
   - Used for responsive UI design.

#### 3. **`Scaffold` (Main Layout)**
   - Provides the basic structure of the app screen.
   - Uses a `Stack` widget to layer different UI elements.

#### 4. **`_featuredWallPapers()`**
   - Displays a horizontally scrollable list of wallpapers using `PageView`.
   - Updates `_selectedWallPaper` when a new wallpaper is selected.

#### 5. **`_gradientBoxWidget()`**
   - Creates a gradient overlay from dark to transparent.
   - Positioned at the bottom using `Align`.

#### 6. **`_topLayerWidget()`**
   - Contains various UI elements such as:
     - `_topBarWidget()` (App bar with icons)
     - `_featuredWallPapersInfoWidget()` (Wallpaper title and indicator)
     - `ScrollableWallPaperWidget()` (Horizontally scrolling wallpaper list)
     - `_featuredWallPaperBannerWidget()` (A featured wallpaper banner)

#### 7. **`_topBarWidget()`**
   - A row with icons for menu, search, and notifications.

#### 8. **`_featuredWallPapersInfoWidget()`**
   - Displays the title of the selected wallpaper.
   - Shows a row of small circular indicators to highlight the current wallpaper.

#### 9. **`_featuredWallPaperBannerWidget()`**
   - A banner displaying a featured wallpaper with an image from the `featuredWallPapers` list.


## widgets/scrollable_wallpager_widget.dart

```dart
import 'package:flutter/material.dart';
import 'package:wally/wallpapers.dart';

class ScrollableWallPaperWidget extends StatelessWidget {
  final double _height;
  final double _width;
  final bool _isShowTitle;
  final List<WallPaper> _wallPaperData;

  ScrollableWallPaperWidget(
    this._height,
    this._width,
    this._isShowTitle,
    this._wallPaperData,
  );

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: _height,
      width: _width,
      child: ListView(
        physics: BouncingScrollPhysics(),
        scrollDirection: Axis.horizontal,
        children:
            _wallPaperData.map((_wallPaper) {
              return Container(
                height: _height,
                width: _width*0.3,
                padding: EdgeInsets.only(right: _width * 0.03),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  mainAxisSize: MainAxisSize.max,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Container(
                      height: _height * 0.70,
                      width: _width * 0.45,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(5),
                        image: DecorationImage(
                          fit: BoxFit.cover,
                          image: NetworkImage(_wallPaper.coverImage.url),
                        ),
                      ),
                    ),
                    _isShowTitle?Text(
                      _wallPaper.title,
                      maxLines: 2,
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: _height * 0.08,
                      ),
                    ):Container(),
                  ],
                ),
              );
            }).toList(),
      ),
    );
  }
}
```

### Explanation of Components and Widgets

#### 1. **`ScrollableWallPaperWidget` (Stateless Widget)**
   - A horizontally scrollable list of wallpapers.
   - Takes four parameters:
     - `_height`: The height of the widget.
     - `_width`: The width of the widget.
     - `_isShowTitle`: Boolean to decide if the wallpaper title should be displayed.
     - `_wallPaperData`: List of wallpapers.

#### 2. **`SizedBox` (Container for ListView)**
   - Defines the size of the scrollable widget using `_height` and `_width`.

#### 3. **`ListView` (Scrollable Wallpaper List)**
   - Uses `BouncingScrollPhysics()` for a smooth scrolling effect.
   - Scrolls horizontally (`scrollDirection: Axis.horizontal`).

#### 4. **Mapping `_wallPaperData` to UI**
   - Iterates through `_wallPaperData` to create wallpaper cards.
   - Each wallpaper is wrapped in a `Container` with:
     - Fixed height and width (`_height`, `_width * 0.3`).
     - Right padding for spacing.

#### 5. **Wallpaper Display (`Container` with `BoxDecoration`)**
   - A sub-container with:
     - Fixed height (`_height * 0.70`).
     - Fixed width (`_width * 0.45`).
     - Rounded corners (`borderRadius: BorderRadius.circular(5)`).
     - `DecorationImage` to load the wallpaper using `NetworkImage`.

#### 6. **Wallpaper Title Display (`Text` Widget)**
   - Displays the wallpaper title if `_isShowTitle` is `true`.
   - Uses a condition: `_isShowTitle ? Text(...) : Container()`.
   - The font size is proportional to `_height * 0.08`.
   - Title is white for visibility on dark backgrounds.
