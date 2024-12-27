import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Container } from 'react-bootstrap';
import { discoverMovies } from './actions/discoverMovies';
import { fetchPopularMovies } from './actions/fetchPopularMovies';
import { fetchGenreMovies } from './actions/fetchGenreMovies';
import Header from './components/Header';
import MovieCarousel from './components/MovieCarousel';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../main/main.css';

const GENRES = [
    { id: 28, name: 'Action' },
    { id: 12, name: 'Adventure' },
    { id: 35, name: 'Comedy' },
    { id: 18, name: 'Drama' },
    { id: 10751, name: 'Family' },
    { id: 10749, name: 'Romance' },
];

const App = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [popularMovies, setPopularMovies] = useState({ results: [] });
    const [discoverMoviesList, setDiscoverMoviesList] = useState({ results: [] });
    const [genreMovies, setGenreMovies] = useState({});
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchAllMovies();
    }, []);

    const fetchAllMovies = async () => {
        try {
            setIsLoading(true);
            const [popularRes, discoverRes] = await Promise.all([
                fetchPopularMovies(),
                discoverMovies(1)
            ]);
            
            setPopularMovies(popularRes.data);
            setDiscoverMoviesList(discoverRes.data);

            // Fetch movies for each genre
            const genreResults = {};
            for (const genre of GENRES) {
                const genreRes = await fetchGenreMovies(genre.id);
                genreResults[genre.id] = genreRes.data;
            }
            setGenreMovies(genreResults);
        } catch (err) {
            setError('Failed to fetch movies');
            console.error('Error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <Header />
            <Container fluid>
                {error && <div className="error alert alert-danger">{error}</div>}
                {isLoading ? (
                    <div className="text-center my-5">
                        <div className="spinner-border" role="status">
                            <span className="visually-hidden">Loading...</span>
                        </div>
                    </div>
                ) : (
                    <>
                        {popularMovies.results.length > 0 && (
                            <MovieCarousel
                                movies={popularMovies.results}
                                title="Most Viewed Movies"
                            />
                        )}

                        {discoverMoviesList.results.length > 0 && (
                            <MovieCarousel
                                movies={discoverMoviesList.results}
                                title="Popular Movies"
                            />
                        )}

                        {GENRES.map(genre => (
                            genreMovies[genre.id]?.results.length > 0 && (
                                <MovieCarousel
                                    key={genre.id}
                                    movies={genreMovies[genre.id].results}
                                    title={`${genre.name} Movies`}
                                />
                            )
                        ))}
                    </>
                )}
            </Container>
        </div>
    );
};

const container = document.getElementById('react');
const root = createRoot(container);
root.render(<App />);

export default App;
