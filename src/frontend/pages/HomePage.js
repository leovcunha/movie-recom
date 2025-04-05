import React from 'react';
import { Container } from 'react-bootstrap';
import MovieCarousel from '../components/MovieCarousel';
import { fetchPopularMovies } from '../actions/fetchPopularMovies';
import { discoverMovies } from '../actions/discoverMovies';
import { fetchGenreMovies } from '../actions/fetchGenreMovies';
import { GENRES } from '../constants/genres';

const HomePage = ({ 
    popularMovies, 
    newReleases, 
    genreMovies, 
    isLoading, 
    error, 
    loadMoreMovies, 
    handleRate,
    setPopularMovies,
    setNewReleases,
    setGenreMovies
}) => (
    <Container>
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
                        onLoadMore={() => loadMoreMovies(fetchPopularMovies, popularMovies, setPopularMovies)}
                        currentPage={popularMovies.page}
                        totalPages={popularMovies.total_pages}
                        onRate={handleRate}
                    />
                )}

                {newReleases.results.length > 0 && (
                    <MovieCarousel
                        movies={newReleases.results}
                        title="New Releases"
                        onLoadMore={() => loadMoreMovies(discoverMovies, newReleases, setNewReleases)}
                        currentPage={newReleases.page}
                        totalPages={newReleases.total_pages}
                        onRate={handleRate}
                    />
                )}

                {GENRES.map(genre => (
                    genreMovies[genre.id]?.results.length > 0 && (
                        <MovieCarousel
                            key={genre.id}
                            movies={genreMovies[genre.id].results}
                            title={`${genre.name} Movies`}
                            onLoadMore={() => loadMoreMovies(fetchGenreMovies, genreMovies[genre.id], setGenreMovies, { genreId: genre.id })}
                            currentPage={genreMovies[genre.id]?.page || 1}
                            totalPages={genreMovies[genre.id]?.total_pages || 1}
                            onRate={handleRate}
                        />
                    )
                ))}
            </>
        )}
    </Container>
);

export default HomePage; 