def bf(planet1, planet2):
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    try:
        idx1 = planets.index(planet1)
        idx2 = planets.index(planet2)
    except ValueError:
        return ()
    if idx1 == idx2:
        return ()
    start, end = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
    return tuple(planets[start+1:end])
