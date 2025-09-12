<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Meet Julius Caesar - Onboarding</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        background: #f5f5f5;
        margin: 0;
        padding: 0;
        text-align: center;
      }
      header {
        background: #222;
        color: #fff;
        padding: 1rem;
      }
      main {
        padding: 2rem;
      }
      button {
        background: #0078d4;
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        cursor: pointer;
        border-radius: 6px;
      }
      button:hover {
        background: #005a9e;
      }
      #status {
        margin-top: 1rem;
        font-weight: bold;
      }
    </style>
  </head>
  <body>
    <header>
      <h1>Welcome to Julius Caesar Onboarding</h1>
    </header>

    <main>
      <p>Click below to start your onboarding process.</p>
      <button id="startBtn">Join with Caesar</button>
      <div id="status" aria-live="polite"></div>
    </main>

    <script>
      const CONFIG = {
        apiBase: "https://YOUR-BACKEND-API-URL", // e.g., https://api.yourngo.org/caesar
        mailFrom: "placeholder@example.com" // temporary until NGO email is ready
      };

      const startBtn = document.getElementById("startBtn");
      const statusEl = document.getElementById("status");

      startBtn.addEventListener("click", async () => {
        statusEl.textContent = "Starting onboarding...";

        try {
          const res = await fetch(`${CONFIG.apiBase}/join/provision`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              email: CONFIG.mailFrom
            })
          });

          if (!res.ok) {
            const msg = `HTTP ${res.status}`;
            throw new Error(msg);
          }

          const data = await res.json();
          console.log("Response:", data);
          statusEl.textContent = "Onboarding started successfully.";
        } catch (err) {
          console.error(err);
          statusEl.textContent = "Error starting onboarding. Please try again.";
        }
      });
    </script>
  </body>
</html>