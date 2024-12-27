import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { discoverMovies } from './actions/discoverMovies';
import { fetchPopularMovies } from './actions/fetchPopularMovies';
import { fetchGenreMovies } from './actions/fetchGenreMovies';
import Header from './components/Header';
import MovieTable from './components/MovieTable';
import '../main/main.css';

const GENRES = [
    { id: 28, name: 'Action' },
    { id: 12, name: 'Adventure' },
    { id: 35, name: 'Comedy' },
    { id: 18, name: 'Drama' }
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
            {error && <div className="error">{error}</div>}
            {isLoading ? (
                <div>Loading...</div>
            ) : (
                <>
                    {popularMovies.results.length > 0 && (
                        <div className="movie-section">
                            <h2>Most Viewed Movies</h2>
                            <MovieTable
                                movies={popularMovies.results}
                                pagehandler={() => {}}
                                currentPage={1}
                            />
                        </div>
                    )}

                    {discoverMoviesList.results.length > 0 && (
                        <div className="movie-section">
                            <h2>Popular Movies</h2>
                            <MovieTable
                                movies={discoverMoviesList.results}
                                pagehandler={() => {}}
                                currentPage={1}
                            />
                        </div>
                    )}

                    {GENRES.map(genre => (
                        genreMovies[genre.id]?.results.length > 0 && (
                            <div key={genre.id} className="movie-section">
                                <h2>Popular {genre.name} Movies</h2>
                                <MovieTable
                                    movies={genreMovies[genre.id].results}
                                    pagehandler={() => {}}
                                    currentPage={1}
                                />
                            </div>
                        )
                    ))}
                </>
            )}
        </div>
    );
};

const container = document.getElementById('react');
const root = createRoot(container);
root.render(<App />);

export default App;
