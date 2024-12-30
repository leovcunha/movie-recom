import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { discoverMovies } from './actions/discoverMovies';
import { fetchPopularMovies } from './actions/fetchPopularMovies';
import { fetchGenreMovies } from './actions/fetchGenreMovies';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import RecommendationsPage from './pages/RecommendationsPage';
import { GENRES } from './constants/genres';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../main/main.css';

const App = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [popularMovies, setPopularMovies] = useState({ results: [], page: 1 });
    const [newReleases, setNewReleases] = useState({ results: [], page: 1 });
    const [genreMovies, setGenreMovies] = useState({});
    const [error, setError] = useState(null);
    const [showRecommendations, setShowRecommendations] = useState(false);
    const [userRatings, setUserRatings] = useState({});

    useEffect(() => {
        fetchAllMovies();
        // Load saved ratings from localStorage
        const savedRatings = JSON.parse(localStorage.getItem('userRatings') || '{}');
        setUserRatings(savedRatings);
        setShowRecommendations(Object.keys(savedRatings).length >= 5);
    }, []);

    const loadMoreMovies = async (fetchFunction, currentState, setState, params = {}) => {
        try {
            const nextPage = currentState.page + 1;
            const response = await fetchFunction(params.genreId ? params.genreId : nextPage);
            
            if (params.genreId) {
                setState(prev => ({
                    ...prev,
                    [params.genreId]: {
                        ...response.data,
                        results: [...prev[params.genreId].results, ...response.data.results].map(movie => ({
                            ...movie,
                            userRating: userRatings[movie.id] || 0
                        })),
                        page: nextPage,
                        total_pages: response.data.total_pages
                    }
                }));
            } else {
                setState(prev => ({
                    ...response.data,
                    results: [...prev.results, ...response.data.results].map(movie => ({
                        ...movie,
                        userRating: userRatings[movie.id] || 0
                    })),
                    page: nextPage
                }));
            }
        } catch (err) {
            console.error('Error loading more movies:', err);
        }
    };

    const handleRate = (movieId, rating) => {
        const newRatings = { ...userRatings, [movieId]: rating };
        setUserRatings(newRatings);
        localStorage.setItem('userRatings', JSON.stringify(newRatings));
        
        // Show recommendations tab if user has rated at least 5 movies
        if (Object.keys(newRatings).length >= 5) {
            setShowRecommendations(true);
        }

        // Update movie ratings in all lists
        updateMovieRatings(movieId, rating);
    };

    const updateMovieRatings = (movieId, rating) => {
        // Update popular movies
        setPopularMovies(prev => ({
            ...prev,
            results: prev.results.map(movie => 
                movie.id === movieId ? { ...movie, userRating: rating } : movie
            )
        }));

        // Update new releases
        setNewReleases(prev => ({
            ...prev,
            results: prev.results.map(movie => 
                movie.id === movieId ? { ...movie, userRating: rating } : movie
            )
        }));

        // Update genre movies
        setGenreMovies(prev => {
            const updated = { ...prev };
            Object.keys(updated).forEach(genreId => {
                updated[genreId] = {
                    ...updated[genreId],
                    results: updated[genreId].results.map(movie => 
                        movie.id === movieId ? { ...movie, userRating: rating } : movie
                    )
                };
            });
            return updated;
        });
    };

    const fetchAllMovies = async () => {
        try {
            setIsLoading(true);
            const [popularRes, newReleasesRes] = await Promise.all([
                fetchPopularMovies(1),
                discoverMovies(1)
            ]);
            
            // Add user ratings to movies
            const popularWithRatings = {
                ...popularRes.data,
                results: popularRes.data.results.map(movie => ({
                    ...movie,
                    userRating: userRatings[movie.id] || 0
                }))
            };

            const newReleasesWithRatings = {
                ...newReleasesRes.data,
                results: newReleasesRes.data.results.map(movie => ({
                    ...movie,
                    userRating: userRatings[movie.id] || 0
                }))
            };

            setPopularMovies(popularWithRatings);
            setNewReleases(newReleasesWithRatings);

            // Fetch movies for each genre
            const genreResults = {};
            for (const genre of GENRES) {
                const genreRes = await fetchGenreMovies(genre.id);
                genreResults[genre.id] = {
                    ...genreRes.data,
                    results: genreRes.data.results.map(movie => ({
                        ...movie,
                        userRating: userRatings[movie.id] || 0
                    }))
                };
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
        <Router>
            <div className="app-container">
                <Header showRecommendations={showRecommendations} />
                <Routes>
                    <Route 
                        path="/" 
                        element={
                            <HomePage 
                                popularMovies={popularMovies}
                                newReleases={newReleases}
                                genreMovies={genreMovies}
                                isLoading={isLoading}
                                error={error}
                                loadMoreMovies={loadMoreMovies}
                                handleRate={handleRate}
                                setPopularMovies={setPopularMovies}
                                setNewReleases={setNewReleases}
                                setGenreMovies={setGenreMovies}
                            />
                        } 
                    />
                    <Route 
                        path="/recommendations" 
                        element={<RecommendationsPage onRate={handleRate} />} 
                    />
                </Routes>
            </div>
        </Router>
    );
};

const container = document.getElementById('react');
const root = createRoot(container);
root.render(<App />);

export default App;
