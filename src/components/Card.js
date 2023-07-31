import React from "react";

function Card(props) {
  return(
    <div class="card card-compact w-65 bg-base-100 shadow-xl" style={{ zIndex: -10 }}>
      <h1>{props.imgPath}</h1>
      <figure><img src={props.imageSrc} alt="Dataset Sample" /></figure>
      <div class="card-body" style={{ zIndex: -10 }}>
        <h2 class="card-title">{props.title}</h2>
        <p>{props.description}</p>
      </div>
    </div>
  )
}export default Card;