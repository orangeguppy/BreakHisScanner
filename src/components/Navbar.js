import React from "react";
import {Link} from "react-router-dom";

function Navbar() {
    return(
      <div class="navbar bg-base-100">
        <div class="flex-1">
          <Link to="/" class="btn btn-ghost normal-case text-xl">BreakHis Scanner</Link>
        </div>
        <div class="flex-none">
          <ul class="menu menu-horizontal px-1">
            <li><Link to="/">Tool</Link></li>
            <li tabindex="0">
              <a>
                Documentation
                <svg class="fill-current" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z"/></svg>
              </a>
              <ul class="p-2 bg-base-100">
                <li><Link to="/plan">Project Plan</Link></li>
                <li><Link to="/eda">Exploratory Data Analysis</Link></li>
                <li><Link to="/data-prep">Data Preparation</Link></li>
                <li><a>Training and Testing</a></li>
                <li><a>Model Evauation</a></li>
              </ul>
            </li>
            <li><a>Feedback</a></li>
          </ul>
        </div>
      </div>
    )
}
export default Navbar;