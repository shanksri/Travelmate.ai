const form = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitButton = document.getElementById("submit-button");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const chipsEl = document.getElementById("chips");
const includeFlights = document.getElementById("include-flights");
const includeHotels = document.getElementById("include-hotels");
const reviseCard = document.getElementById("revise-card");
const reviseForm = document.getElementById("revise-form");
const reviseInput = document.getElementById("revise-input");
const reviseButton = document.getElementById("revise-button");

// The trip every change applies to. Each change is saved server-side as the
// next version of this thread, so earlier versions are never overwritten.
let currentThreadId = null;

checkOnlineStatus();

chipsEl.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip) return;
  promptInput.value = chip.dataset.prompt;
  promptInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  resultEl.hidden = true;
  reviseCard.hidden = true;
  submitButton.disabled = true;
  setStatus(
    "loading",
    "Planning your trip — this makes several real OpenAI calls and can take 30-60 seconds..."
  );

  try {
    const response = await fetch("/trips/plan-from-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        include_flights: includeFlights.checked,
        include_hotels: includeHotels.checked,
      }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(extractErrorMessage(data));
    }

    hideStatus();
    renderTrip(data.trip);
  } catch (err) {
    setStatus("error", err.message || "Something went wrong.");
  } finally {
    submitButton.disabled = false;
  }
});

reviseForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const changeRequest = reviseInput.value.trim();
  if (!changeRequest || !currentThreadId) return;

  reviseButton.disabled = true;
  submitButton.disabled = true;
  setStatus("loading", "Updating your plan — this rewrites the itinerary and can take 30-60 seconds...");

  try {
    const response = await fetch(`/trips/${encodeURIComponent(currentThreadId)}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ change_request: changeRequest }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(extractErrorMessage(data));
    }

    hideStatus();
    reviseInput.value = "";
    renderTrip(data.trip);
  } catch (err) {
    // The plan on screen is untouched, so the change can simply be retried.
    setStatus("error", err.message || "Something went wrong.");
  } finally {
    reviseButton.disabled = false;
    submitButton.disabled = false;
  }
});

async function checkOnlineStatus() {
  try {
    const response = await fetch("/health");
    setOnlineStatus(response.ok);
  } catch {
    setOnlineStatus(false);
  }
}

function setOnlineStatus(isOnline) {
  statusDot.className = `dot ${isOnline ? "online" : "offline"}`;
  statusText.textContent = isOnline ? "Online" : "Offline";
}

function extractErrorMessage(data) {
  if (!data || data.detail === undefined) return "Request failed.";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    // FastAPI's own 422 validation-error shape: a list of {loc, msg, type}.
    return data.detail
      .map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`)
      .join("; ");
  }
  return "Request failed.";
}

function setStatus(kind, message) {
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`;
  statusEl.textContent = message;
}

function hideStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

// Set from the rendered trip's own `currency`, so a trip stored before the
// switch to rupees still renders in the currency it was actually priced in.
let tripCurrency = "INR";

function money(amount) {
  if (amount === null || amount === undefined) return null;
  // en-IN gives rupees their conventional grouping — 1,50,000 rather than 150,000.
  const locale = tripCurrency === "INR" ? "en-IN" : undefined;
  const symbol = tripCurrency === "INR" ? "₹" : "$";
  return `${symbol}${Number(amount).toLocaleString(locale, { maximumFractionDigits: 0 })}`;
}

const OPTIONS_PER_ROW = 3;

function flightDuration(hours) {
  if (hours === null || hours === undefined) return "—";
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return m ? `${h}h ${m}m` : `${h}h`;
}

// Read straight off the ISO string rather than through `new Date(...)`, which
// would shift the time into the viewer's timezone — a 06:00 departure from
// Delhi should read 06:00 wherever the page is opened.
function flightWhen(flight) {
  const iso = flight.departure_at || flight.depart_date;
  if (!iso) return "";
  const day = new Date(`${iso.slice(0, 10)}T00:00:00`).toLocaleDateString("en-IN", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  return flight.departure_at ? `${day}, ${iso.slice(11, 16)}` : day;
}

function renderFlightOption(flight) {
  const stops = flight.stops === 0 ? "nonstop" : `${flight.stops} stop(s)`;
  const price = money(flight.total);
  return `
    <div class="option-row">
      <span class="option-info">
        <span class="option-duration">${flightDuration(flight.duration_hours)}</span>
        <span class="option-stops">${stops}</span>
        <span class="option-meta">${escapeHtml(flightWhen(flight))}</span>
      </span>
      <span class="option-price">${price ?? "—"}</span>
    </div>`;
}

// Cheapest and fastest are picked here rather than server-side so both rows
// come from the one option list the itinerary already carries.
function renderFlightTable(label, options) {
  if (!options || !options.length) return "";

  const route = options[0].origin && options[0].destination
    ? ` — ${escapeHtml(options[0].origin)} → ${escapeHtml(options[0].destination)}`
    : "";
  const byPrice = [...options].sort((a, b) => (a.total ?? Infinity) - (b.total ?? Infinity));
  const byDuration = [...options]
    // A missing duration would otherwise sort to the front and be presented
    // as the fastest flight available.
    .filter((f) => f.duration_hours !== null && f.duration_hours !== undefined)
    .sort((a, b) => a.duration_hours - b.duration_hours);

  const row = (title, list) =>
    list.length
      ? `<tr>
           <th scope="row">${title}</th>
           <td colspan="2">${list.slice(0, OPTIONS_PER_ROW).map(renderFlightOption).join("")}</td>
         </tr>`
      : "";

  return `
    <div class="card">
      <h3>${escapeHtml(label)}${route}</h3>
      <table class="flight-table">
        <thead>
          <tr><th></th><th>Flight</th><th class="price-col">Price</th></tr>
        </thead>
        <tbody>
          ${row("Cheapest", byPrice)}
          ${row("Fastest", byDuration)}
        </tbody>
      </table>
    </div>`;
}

function renderHotelItem(hotel, isSelected) {
  const bits = [];
  if (hotel.tier) bits.push(escapeHtml(hotel.tier));
  if (hotel.rating !== null && hotel.rating !== undefined) bits.push(`${hotel.rating} rating`);
  const tag = bits.length ? ` (${bits.join(", ")})` : "";
  const nightly = money(hotel.nightly);
  const total = money(hotel.total);
  const cost = nightly && total ? ` — ${nightly}/night, ${total} total` : "";
  return `
    <li class="${isSelected ? "selected" : ""}">
      <span><span class="hotel-name">${escapeHtml(hotel.name)}</span>${tag}<span class="hotel-meta">${cost}</span></span>
      ${isSelected ? '<span class="selected-tag">Selected</span>' : ""}
    </li>`;
}

function renderActivity(activity) {
  const where = activity.location ? ` @ ${escapeHtml(activity.location)}` : "";
  return `
    <div class="activity">
      <div class="activity-row">
        <span class="activity-time">${escapeHtml(activity.time)}</span>
        <span class="activity-title">${escapeHtml(activity.title)}${where}</span>
      </div>
      ${activity.description ? `<p class="activity-description">${escapeHtml(activity.description)}</p>` : ""}
    </div>`;
}

function renderDay(day) {
  return `
    <div class="day">
      <h4>Day ${day.day} — ${escapeHtml(day.date)}: ${escapeHtml(day.summary)}</h4>
      ${day.activities.map(renderActivity).join("")}
    </div>`;
}

function renderTrip(trip) {
  const it = trip.itinerary;
  tripCurrency = it.currency || "INR";
  const total = money(it.total_estimated_cost);

  currentThreadId = trip.thread_id || trip.id;
  reviseCard.hidden = false;

  resultEl.hidden = false;
  resultEl.innerHTML = `
    <div class="result-header">
      <h2>Your Trip Plan</h2>
      <span class="trip-id">Trip ID: ${escapeHtml(trip.id)}</span>
    </div>

    <div class="card trip-header">
      <h2>${escapeHtml(it.destination)} — ${escapeHtml(it.start_date)} to ${escapeHtml(it.end_date)}, ${it.travelers} traveller(s)</h2>
      ${total ? `<p class="trip-cost">Estimated total: ${total}</p>` : ""}
      ${trip.summary ? `<p class="summary">${escapeHtml(trip.summary)}</p>` : ""}
    </div>

    ${renderFlightTable("Outbound options", it.outbound_options)}
    ${renderFlightTable("Return options", it.return_options)}

    ${
      it.lodging_options.length
        ? `<div class="card">
            <h3>Where to stay</h3>
            <ul class="hotel-list">
              ${it.lodging_options.map((h, i) => renderHotelItem(h, i === 0)).join("")}
            </ul>
          </div>`
        : ""
    }

    <div class="card">
      <h3>Day by day</h3>
      ${it.days.map(renderDay).join("")}
    </div>
  `;
}
