from __future__ import annotations

from .models import ThemeDefinition


DEFAULT_THEME_CATALOG: tuple[ThemeDefinition, ...] = (
    ThemeDefinition(
        key="noise",
        label="Noise and sleep disruption",
        prototypes=(
            "The room was noisy because of traffic, elevators, hallway noise, thin walls, or loud neighbors.",
            "We could hear everything all night and it was hard to sleep because the hotel was loud.",
            "Street noise, doors slamming, ice machines, and other guests kept us awake.",
        ),
    ),
    ThemeDefinition(
        key="staff_service",
        label="Front desk and service problems",
        prototypes=(
            "The front desk was rude, dismissive, or unhelpful and staff did not resolve the issue.",
            "Customer service was poor and management ignored complaints or treated guests badly.",
            "Reception handled the stay badly and support from staff was disappointing.",
        ),
    ),
    ThemeDefinition(
        key="check_in",
        label="Check-in or reservation issues",
        prototypes=(
            "Check-in took too long, the room was not ready, or arrival was handled poorly.",
            "The reservation was wrong at arrival and the hotel gave the wrong room or no room.",
            "We had problems with check-in, keys, room assignment, or the booking details.",
        ),
    ),
    ThemeDefinition(
        key="cleanliness",
        label="Cleanliness and hygiene problems",
        prototypes=(
            "The room was dirty, filthy, dusty, stained, or clearly not cleaned properly.",
            "There were dirty sheets, hair, trash, grime, or unwashed towels in the room.",
            "The hotel felt unsanitary and hygiene standards were poor.",
        ),
    ),
    ThemeDefinition(
        key="room_condition",
        label="Room condition and maintenance problems",
        prototypes=(
            "The room felt old, damaged, broken down, or badly maintained and needed renovation.",
            "Fixtures, furniture, walls, or equipment were broken, worn out, or in disrepair.",
            "The property had maintenance issues and the room condition was poor.",
        ),
    ),
    ThemeDefinition(
        key="bathroom_plumbing",
        label="Bathroom or plumbing issues",
        prototypes=(
            "The bathroom had leaks, flooding, broken shower parts, toilet issues, or plumbing problems.",
            "Water overflowed, the shower did not work, or the bathroom fixtures were broken.",
            "The bathroom was unusable or unsafe because of plumbing or water problems.",
        ),
    ),
    ThemeDefinition(
        key="temperature_airflow",
        label="Heat, AC, or airflow problems",
        prototypes=(
            "The room was too hot, stuffy, poorly ventilated, or the air conditioning did not work.",
            "Sun exposure made the room overheat and it was hard to cool the room down.",
            "Temperature control was bad and the room felt uncomfortable because of heat or airflow.",
        ),
    ),
    ThemeDefinition(
        key="smell_air_quality",
        label="Bad smells or air quality problems",
        prototypes=(
            "The room smelled moldy, musty, smoky, or otherwise unpleasant.",
            "There was mildew, urine smell, smoke odor, or bad air quality in the room.",
            "A strong smell made the stay uncomfortable and the air felt dirty.",
        ),
    ),
    ThemeDefinition(
        key="bed_comfort",
        label="Bed comfort and sleep comfort problems",
        prototypes=(
            "The bed, pillows, or bedding were uncomfortable and made it hard to rest.",
            "The mattress was bad, sheets were poor, or the sofa bed was uncomfortable.",
            "Sleep comfort was poor because of the bed or bedding quality.",
        ),
    ),
    ThemeDefinition(
        key="location_access",
        label="Location or area problems",
        prototypes=(
            "The location felt unsafe, inconvenient, hard to reach, or far from what we needed.",
            "The neighborhood was bad or the property was difficult to access from the road.",
            "The area around the hotel was disappointing or the location was not good.",
        ),
    ),
    ThemeDefinition(
        key="parking",
        label="Parking and arrival access problems",
        prototypes=(
            "Parking was difficult, too small, unavailable, or hard to get in and out of.",
            "The parking lot was inconvenient, unsafe, or poorly managed.",
            "Arriving by car was frustrating because of parking or driveway access issues.",
        ),
    ),
    ThemeDefinition(
        key="amenities_unavailable",
        label="Missing or unavailable amenities",
        prototypes=(
            "Amenities that were expected were closed, missing, broken, or unavailable.",
            "The pool, hot tub, restaurant, shuttle, gym, or other hotel features were not usable.",
            "Promised amenities were unavailable during the stay.",
        ),
    ),
    ThemeDefinition(
        key="food_breakfast",
        label="Breakfast or food service problems",
        prototypes=(
            "Breakfast was poor, unavailable, disappointing, or handled badly.",
            "Food service was low quality, missing, overpriced, or much worse than expected.",
            "The restaurant or breakfast experience was a problem during the stay.",
        ),
    ),
    ThemeDefinition(
        key="wifi_connectivity",
        label="Wi-Fi or connectivity problems",
        prototypes=(
            "The Wi-Fi was weak, unreliable, slow, or did not work.",
            "Internet access kept failing and the connection was unusable.",
            "Connectivity problems made it hard to use the hotel internet.",
        ),
    ),
    ThemeDefinition(
        key="listing_accuracy",
        label="Listing accuracy or room mismatch problems",
        prototypes=(
            "The hotel was not as advertised and the room or amenities did not match the listing.",
            "We booked one thing and received a different room type, setup, or feature.",
            "The online description over-promised and the actual stay did not match it.",
        ),
    ),
    ThemeDefinition(
        key="value",
        label="Poor value for money",
        prototypes=(
            "The stay felt overpriced and not worth what we paid.",
            "The hotel charged too much for the quality that was delivered.",
            "Value for money was poor and the property was not worth the price.",
        ),
    ),
    ThemeDefinition(
        key="safety",
        label="Safety or security concerns",
        prototypes=(
            "The stay felt unsafe because of hazards, broken features, security issues, or dangerous conditions.",
            "There were safety risks in the room or around the property.",
            "Security and guest safety were a concern during the stay.",
        ),
    ),
    ThemeDefinition(
        key="pests",
        label="Pests or infestation",
        prototypes=(
            "There were bed bugs, roaches, insects, or other pests in the room.",
            "Infestation or bugs made the stay unacceptable.",
            "The room had pests and cleanliness problems because of insects or vermin.",
        ),
    ),
    ThemeDefinition(
        key="smoking",
        label="Smoke exposure or smoking policy problems",
        prototypes=(
            "The room or hotel smelled like cigarettes, weed, or smoke even though it should not have.",
            "Guests were smoking nearby and the smoke reached the room.",
            "Smoke exposure made the stay uncomfortable and the non-smoking policy was not enforced.",
        ),
    ),
    ThemeDefinition(
        key="construction",
        label="Construction or renovation disruption",
        prototypes=(
            "Construction, renovations, or repair work disrupted the stay and made the property inconvenient.",
            "The hotel was under renovation and noise, mess, or closures affected the stay.",
            "Building work created disruption and the property should have disclosed it more clearly.",
        ),
    ),
    ThemeDefinition(
        key="size_layout",
        label="Room size or layout problems",
        prototypes=(
            "The room was too small, awkwardly laid out, or missing expected space.",
            "The layout was inconvenient and the room setup did not work well.",
            "Space, room size, or configuration was a problem during the stay.",
        ),
    ),
)
