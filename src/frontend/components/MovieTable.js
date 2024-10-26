import React, { useState, useEffect } from "react";
import MovieHolder from "./MovieHolder";
import { discoverMovies } from "../actions/discoverMovies";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faChevronLeft, faChevronRight } from "@fortawesome/free-solid-svg-icons";

const MovieTable = ({ movies, pagehandler, currentPage }) => {
    const handleClick = (e, direction) => {
        e.preventDefault();
        console.log(`${direction} arrow clicked`);
        console.log("Current page:", currentPage);
        pagehandler(direction === "right");
    };

    return (
        <div className="movietable">
            <div
                className="arrow-left"
                onClick={(e) => handleClick(e, "left")}
                style={{ cursor: "pointer" }}
            >
                <FontAwesomeIcon icon={faChevronLeft} size="2x" />
            </div>
            <div className="movielist">
                {movies?.map((movieData) => (
                    <MovieHolder key={movieData.id} movie={movieData} />
                ))}
            </div>
            <div
                className="arrow-right"
                onClick={(e) => handleClick(e, "right")}
                style={{ cursor: "pointer" }}
            >
                <FontAwesomeIcon icon={faChevronRight} size="2x" />
            </div>
        </div>
    );
};

export default MovieTable;
