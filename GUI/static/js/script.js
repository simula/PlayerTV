document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("videoForm").addEventListener("submit", function (e) {
    e.preventDefault();

    var m3u8Url = document.getElementById("m3u8Input").value;

    // Send a POST request to the server to process the video
    fetch("/process_video", {
      method: "POST",
      body: JSON.stringify({ m3u8Url: m3u8Url }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.mp4Path) {
          var videoPlayer = document.getElementById("videoPlayer");
          videoPlayer.src = data.mp4Path;
          // Set the video to muted before playing
          videoPlayer.muted = true;
          videoPlayer.load();
          videoPlayer.play();
        } else {
          console.error("Failed to process video");
        }
      })
      .catch((error) => console.error("Error:", error));
  });

  // Pop up functionality
  var configModal = document.getElementById("configAdvModal");
  var configBtn = document.getElementById("configAdvMode");
  var configSpan = document.getElementsByClassName("config-modal-close")[0];

  // Toggle modal display based on checkbox state
  configBtn.addEventListener("change", function () {
    if (this.checked) {
      configModal.style.display = "block";
    } else {
      configModal.style.display = "none";
    }
  });

  // When clicking on the modal close button, uncheck the checkbox and hide the modal
  configSpan.onclick = function () {
    configModal.style.display = "none";
    configBtn.checked = false;
  };

  // When clicking outside the modal, uncheck the checkbox and hide the modal
  window.onclick = function (event) {
    if (event.target == configModal) {
      configModal.style.display = "none";
      configBtn.checked = false;
    }
  };

  // Toggle functionality for Input Section
  var accordionToggleVideo = document.getElementById("accordionToggleVideo");
  var videoSection = document.getElementById("videoSection");

  if (accordionToggleVideo && videoSection) {
    accordionToggleVideo.addEventListener("click", function () {
      videoSection.classList.toggle("hidden");
      var icon = this.querySelector("i");

      // Toggle between fa-angle-down and fa-angle-up
      if (icon.classList.contains("fa-angle-down")) {
        icon.classList.remove("fa-angle-down");
        icon.classList.add("fa-angle-up");
      } else {
        icon.classList.remove("fa-angle-up");
        icon.classList.add("fa-angle-down");
      }
    });
  } else {
    console.error("Accordion elements for video not found");
  }

  // Toggle functionality for Output Section 3
  var accordionToggleVideo2 = document.getElementById("accordionToggleVideo2");
  var filteredResults = document.getElementById("filteredResults");

  if (accordionToggleVideo2 && filteredResults) {
    accordionToggleVideo2.addEventListener("click", function () {
      filteredResults.classList.toggle("hidden");
      var icon = this.querySelector("i");

      // Toggle between fa-angle-down and fa-angle-up
      if (icon.classList.contains("fa-angle-down")) {
        icon.classList.remove("fa-angle-down");
        icon.classList.add("fa-angle-up");
      } else {
        icon.classList.remove("fa-angle-up");
        icon.classList.add("fa-angle-down");
      }
    });
  } else {
    console.error("Accordion elements for filtered results not found");
  }

  // Toggle functionality for Configuration Section 2
  var accordionToggleVideo3 = document.getElementById("accordionToggleVideo3");
  var configurationSection = document.getElementById("configurationSection");

  if (accordionToggleVideo3 && configurationSection) {
    accordionToggleVideo3.addEventListener("click", function () {
      configurationSection.classList.toggle("hidden");
      var icon = this.querySelector("i");

      // Toggle between fa-angle-down and fa-angle-up
      if (icon.classList.contains("fa-angle-down")) {
        icon.classList.remove("fa-angle-down");
        icon.classList.add("fa-angle-up");
      } else {
        icon.classList.remove("fa-angle-up");
        icon.classList.add("fa-angle-down");
      }
    });
  } else {
    console.error("Accordion elements for filtered results not found");
  }
});

// Assuming the base path for team images
const basePathForTeamImages = "/static/images-team-jerseys/2023-jerseys/";

// Function to update team images based on the selected team
function updateTeamImages(selectedTeam) {
  const imageUrl = basePathForTeamImages + selectedTeam + ".jpg"; // Assuming the images are in JPG format
  const imageUrl2 = basePathForTeamImages + selectedTeam + "1.jpg";

  console.log(imageUrl);
  // Set the image URLs
  document.getElementById("leftTeamImage").src = imageUrl;
  document.getElementById("leftTeamImage2").src = imageUrl2;
}

// Assuming the base path for team images
const basePathForPlayerImages = "/static/images-players/";

// Function to update player images based on the selected player
function updatePlayerImages(selectedPlayer, selectedTeam) {
  const plyaerImageUrl = basePathForPlayerImages + selectedTeam + selectedPlayer + ".png";

  console.log(plyaerImageUrl);
  // Set the image URLs

  document.getElementById("rightTeamImage").src = plyaerImageUrl;
}

document.addEventListener("DOMContentLoaded", function () {
  const teamDropdown = document.getElementById("teamDropdown");
  const playerDropdown = document.getElementById("playerDropdown");
  let jsonData = null;

  // Event listener for file input change
  document.getElementById("fileInput").addEventListener("change", function (e) {
    document.getElementById("JersyContainer").style.display = "block";
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (fileEvent) {
      try {
        jsonData = JSON.parse(fileEvent.target.result);
        populateTeamDropdown(jsonData); // Populate team dropdown
        // Initially populate player dropdown with first team's players or empty
        if (teamDropdown.options.length > 0) {
          populatePlayerDropdown(jsonData, teamDropdown.options[0].value);
        }
      } catch (error) {
        console.error("Error parsing JSON:", error);
        alert("Invalid JSON file.");
      }
    };
    reader.readAsText(file);

    // Update team images when the team dropdown is populated
    if (teamDropdown.options.length > 0) {
      updateTeamImages(teamDropdown.options[0].value);
    }
  });

  // Event listener for team dropdown change
  teamDropdown.addEventListener("change", function (e) {
    const selectedTeam = e.target.value;
    populatePlayerDropdown(jsonData, selectedTeam);

    // Update team images when the team selection changes
    updateTeamImages(selectedTeam);
  });

  // Event listener for player dropdown change
  playerDropdown.addEventListener("change", function (e) {
    const selectedPlayer = e.target.value;

    // Update team images when the team selection changes
    updatePlayerImages(selectedPlayer);
  });

  // Event listener for filter button click
  document
    .getElementById("filterButton")
    .addEventListener("click", function () {
      const fileInput = document.getElementById("fileInput");
      const file = fileInput.files[0];
      if (!file) {
        alert("Please select a file.");
        return;
      }
      const team = teamDropdown.value;
      const playerId = playerDropdown.value;
      const formData = new FormData();
      formData.append("file", file);
      formData.append("team", team);
      formData.append("playerId", playerId);

      // Show the overlay as soon as the processing starts
      var element = document.getElementById("filteredResults");
      element.classList.remove("hidden");
      element.scrollIntoView({ behavior: "smooth", block: "start" });
      document.getElementById("overlay_result").style.display = "flex";

      fetch("/upload_and_filter", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.videoPath) {
            var videoSource = document.getElementById("videoSource");
            // Append timestamp to the video URL to prevent caching
            var timestamp = new Date().getTime();
            videoSource.src = data.videoPath + "?t=" + timestamp;
            var videoPlayerResult =
              document.getElementById("videoPlayer_result");

            videoPlayerResult.load(); // Load the new video source

            videoPlayerResult.onloadeddata = function () {
              document.getElementById("overlay_result").style.display = "none";
              videoPlayerResult.play().catch((error) => {
                console.error(
                  "Error occurred while trying to play the video:",
                  error
                );
              });
            };
          } else {
            console.error("Video path not received");
          }
        })
        .catch((error) => console.error("Error:", error));
    });
});

// Function to populate the player dropdown with kit numbers based on the selected team
function populatePlayerDropdown(jsonData, selectedTeam) {
  const kitNumbers = new Set();

  // Iterate through the 'frame' object in the JSON and filter by team
  Object.values(jsonData.frame).forEach((frameData) => {
    Object.values(frameData).forEach((trackData) => {
      if (
        trackData.hasOwnProperty("kit_number") &&
        trackData["team_id"] === selectedTeam
      ) {
        kitNumbers.add(trackData["kit_number"].toString().trim());
      }
    });
  });

  const playerDropdown = document.getElementById("playerDropdown");
  playerDropdown.innerHTML = "";
  kitNumbers.forEach((number) => {
    const option = new Option(number, number);
    playerDropdown.appendChild(option);
  });
}

// Function to populate the team dropdown with team IDs
function populateTeamDropdown(jsonData) {
  const teamIds = new Set();

  // Iterate through the 'frame' object in the JSON
  Object.values(jsonData.frame).forEach((frameData) => {
    // Iterate through each track_id in the frame
    Object.values(frameData).forEach((trackData) => {
      if (trackData.hasOwnProperty("team_id")) {
        teamIds.add(trackData["team_id"].toString().trim());
      }
    });
  });

  const teamDropdown = document.getElementById("teamDropdown");
  teamDropdown.innerHTML = "";
  teamIds.forEach((teamId) => {
    const option = new Option(teamId, teamId);
    teamDropdown.appendChild(option);
  });
}

document.getElementById("videoForm").addEventListener("submit", function (e) {
  e.preventDefault();
  var m3u8Url = document.getElementById("m3u8Input").value;
  document.getElementById("overlay").style.display = "flex";

  fetch("/process_video", {
    method: "POST",
    body: JSON.stringify({ m3u8Url: m3u8Url }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("overlay").style.display = "none"; // Hide the overlay
      if (data.mp4Path) {
        var videoPlayer = document.getElementById("videoPlayer");
        videoPlayer.src = data.mp4Path;
        videoPlayer.load();
        videoPlayer.play();
      } else {
        console.error("Failed to process video");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      document.getElementById("overlay").style.display = "none";
    });
});

// Dark Mode
document
  .getElementById("darkModeToggle")
  .addEventListener("change", function () {
    document.body.classList.toggle("dark-mode", this.checked);
  });
