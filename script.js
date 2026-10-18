const API_URL = "https://smart-cardiopulmacare-1.onrender.com";



// Save the real API response and uploaded audio, then open the dedicated result page.
async function openResultPage(model, data, files) {
    const audioFiles = [];
    for (const file of files) {
        let audioData = "";
        try {
            audioData = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = () => resolve(String(reader.result || ""));
                reader.onerror = () => resolve("");
                reader.readAsDataURL(file);
            });
        } catch (_) {}
        audioFiles.push({ name: file.name, type: file.type || "audio/wav", data: audioData });
    }
    const result = { model, data, files: audioFiles, createdAt: new Date().toLocaleString() };
    try { sessionStorage.setItem("cardioPredictionResult", JSON.stringify(result)); }
    catch (error) {
        // If audio is too large for session storage, keep the actual prediction and filenames.
        result.files = audioFiles.map(file => ({ name: file.name, type: file.type, data: "" }));
        sessionStorage.setItem("cardioPredictionResult", JSON.stringify(result));
    }
    window.location.href = "result.html";
}

// =====================================================
// DOM ELEMENTS
// =====================================================

// Server
const serverDot = document.getElementById("serverDot");
const serverStatus = document.getElementById("serverStatus");


// =====================================================
// MURMUR - 4 VALVE
// =====================================================

const avAudioFile = document.getElementById("avAudioFile");
const mvAudioFile = document.getElementById("mvAudioFile");
const pvAudioFile = document.getElementById("pvAudioFile");
const tvAudioFile = document.getElementById("tvAudioFile");

const avFileName = document.getElementById("avFileName");
const mvFileName = document.getElementById("mvFileName");
const pvFileName = document.getElementById("pvFileName");
const tvFileName = document.getElementById("tvFileName");

const analyzeHeartButton =
    document.getElementById("analyzeHeartButton");

const heartLoading =
    document.getElementById("heartLoading");

const heartResult =
    document.getElementById("heartResult");

const heartResultTitle =
    document.getElementById("heartResultTitle");

const heartProbabilities =
    document.getElementById("heartProbabilities");

const heartError =
    document.getElementById("heartError");


// =====================================================
// AORTIC STENOSIS - AS
// =====================================================

const asAudioFile =
    document.getElementById("asAudioFile");

const asFileName =
    document.getElementById("asFileName");

const analyzeASButton =
    document.getElementById("analyzeASButton");

const asLoading =
    document.getElementById("asLoading");

const asResult =
    document.getElementById("asResult");

const asResultTitle =
    document.getElementById("asResultTitle");

const asResultMessage =
    document.getElementById("asResultMessage");

const asError =
    document.getElementById("asError");


// =====================================================
// MITRAL REGURGITATION - MR
// =====================================================

const mrAudioFile =
    document.getElementById("mrAudioFile");

const mrFileName =
    document.getElementById("mrFileName");

const analyzeMRButton =
    document.getElementById("analyzeMRButton");

const mrLoading =
    document.getElementById("mrLoading");

const mrResult =
    document.getElementById("mrResult");

const mrResultTitle =
    document.getElementById("mrResultTitle");

const mrResultMessage =
    document.getElementById("mrResultMessage");

const mrError =
    document.getElementById("mrError");


// =====================================================
// JSON RESPONSE
// =====================================================

async function getJSONResponse(response) {

    const text = await response.text();

    try {

        return JSON.parse(text);

    } catch (error) {

        throw new Error(
            text || `Server returned ${response.status}`
        );
    }
}


// =====================================================
// MURMUR BUTTON CHECK
// =====================================================

function updateMurmurButton() {
    if (!analyzeHeartButton) return;

    const inputs = [
        { input: avAudioFile, suffix: "AV" },
        { input: mvAudioFile, suffix: "MV" },
        { input: pvAudioFile, suffix: "PV" },
        { input: tvAudioFile, suffix: "TV" }
    ];
    const status = document.getElementById("heartFileStatus");
    const selected = inputs.map(item => {
        const file = item.input && item.input.files && item.input.files[0];
        if (!file) return null;
        const match = file.name.match(new RegExp("^(.+)_" + item.suffix + "\\.wav$", "i"));
        return match ? { patient: match[1].toLowerCase(), file } : null;
    });

    const allSelected = selected.every(Boolean);
    const samePatient = allSelected && selected.every(item => item.patient === selected[0].patient);
    const ready = allSelected && samePatient;
    analyzeHeartButton.disabled = !ready;

    if (status) {
        if (!allSelected) {
            status.textContent = "Upload all 4 correct WAV files: same patient ID with _AV, _MV, _PV and _TV suffixes.";
        } else if (!samePatient) {
            status.textContent = "The 4 files must belong to the same patient. Please check the patient IDs.";
        } else {
            status.textContent = "All 4 valve files verified for the same patient. You can analyze now.";
        }
    }
}


// =====================================================
// FILE INPUT SETUP
// =====================================================

function setupFileInput(
    input,
    nameElement,
    errorElement
) {

    if (!input) return;

    input.addEventListener("change", function () {

        if (!input.files || input.files.length === 0) {

            if (nameElement) {
                nameElement.textContent =
                    "No file selected";
            }

            updateMurmurButton();

            return;
        }


        const file = input.files[0];

        // Verify the valve suffix on Murmur page, so the wrong valve file is rejected.
        const expectedSuffix = {
            avAudioFile: "AV", mvAudioFile: "MV",
            pvAudioFile: "PV", tvAudioFile: "TV"
        }[input.id];
        if (expectedSuffix && !new RegExp("^.+_" + expectedSuffix + "\\.wav$", "i").test(file.name)) {
            if (nameElement) nameElement.textContent = "Wrong valve file";
            if (errorElement) {
                errorElement.textContent = `Please choose the correct ${expectedSuffix} file (example: 2530_${expectedSuffix}.wav). All four files must use the same patient ID.`;
                errorElement.classList.remove("hidden");
            }
            input.value = "";
            updateMurmurButton();
            return;
        }

        // ---------------------------------------------
        // WAV CHECK
        // ---------------------------------------------

        if (
            !file.name
                .toLowerCase()
                .endsWith(".wav")
        ) {

            if (nameElement) {
                nameElement.textContent =
                    "Invalid file";
            }

            if (errorElement) {

                errorElement.textContent =
                    "Please select a WAV audio file.";

                errorElement.classList.remove(
                    "hidden"
                );
            }


            input.value = "";


            // Disable correct button
            if (input.id === "asAudioFile") {

                if (analyzeASButton) {
                    analyzeASButton.disabled = true;
                }
            }


            if (input.id === "mrAudioFile") {

                if (analyzeMRButton) {
                    analyzeMRButton.disabled = true;
                }
            }


            updateMurmurButton();

            return;
        }


        // ---------------------------------------------
        // SHOW FILE NAME
        // ---------------------------------------------

        if (nameElement) {

            nameElement.textContent =
                file.name;
        }


        // Hide error
        if (errorElement) {

            errorElement.classList.add(
                "hidden"
            );
        }


        // ---------------------------------------------
        // AS BUTTON
        // ---------------------------------------------

        if (input.id === "asAudioFile") {

            if (analyzeASButton) {

                analyzeASButton.disabled =
                    false;
            }
        }


        // ---------------------------------------------
        // MR BUTTON
        // ---------------------------------------------

        if (input.id === "mrAudioFile") {

            if (analyzeMRButton) {

                analyzeMRButton.disabled =
                    false;
            }
        }


        // ---------------------------------------------
        // MURMUR BUTTON
        // ---------------------------------------------

        updateMurmurButton();

    });
}


// =====================================================
// SETUP FILE INPUTS
// =====================================================

// Murmur
setupFileInput(
    avAudioFile,
    avFileName,
    heartError
);

setupFileInput(
    mvAudioFile,
    mvFileName,
    heartError
);

setupFileInput(
    pvAudioFile,
    pvFileName,
    heartError
);

setupFileInput(
    tvAudioFile,
    tvFileName,
    heartError
);


// AS
setupFileInput(
    asAudioFile,
    asFileName,
    asError
);


// MR
setupFileInput(
    mrAudioFile,
    mrFileName,
    mrError
);


// =====================================================
// MURMUR ANALYSIS
// =====================================================

if (analyzeHeartButton) {

    analyzeHeartButton.addEventListener(
        "click",
        async function () {


            // -----------------------------------------
            // CHECK 4 FILES
            // -----------------------------------------

            if (
                !avAudioFile.files.length ||
                !mvAudioFile.files.length ||
                !pvAudioFile.files.length ||
                !tvAudioFile.files.length
            ) {

                if (heartError) {

                    heartError.textContent =
                        "Please select all 4 valve WAV files.";

                    heartError.classList.remove(
                        "hidden"
                    );
                }

                return;
            }


            // Disable button
            analyzeHeartButton.disabled = true;


            // Show loading
            if (heartLoading) {

                heartLoading.classList.remove(
                    "hidden"
                );
            }


            // Hide old result
            if (heartResult) {

                heartResult.classList.add(
                    "hidden"
                );
            }


            // Hide error
            if (heartError) {

                heartError.classList.add(
                    "hidden"
                );
            }


            try {

                // -------------------------------------
                // FORM DATA
                // -------------------------------------

                const formData =
                    new FormData();


                // IMPORTANT:
                // Backend expects these exact names
                formData.append(
                    "av_file",
                    avAudioFile.files[0]
                );

                formData.append(
                    "mv_file",
                    mvAudioFile.files[0]
                );

                formData.append(
                    "pv_file",
                    pvAudioFile.files[0]
                );

                formData.append(
                    "tv_file",
                    tvAudioFile.files[0]
                );


                // -------------------------------------
                // CORRECT BACKEND ENDPOINT
                // -------------------------------------

                const response =
                    await fetch(
                        `${API_URL}/predict-patient-heart`,
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await getJSONResponse(
                        response
                    );


                // -------------------------------------
                // ERROR
                // -------------------------------------

                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        data.message ||
                        `Server error: ${response.status}`
                    );
                }


                await openResultPage("Heart Murmur", data, [
                    avAudioFile.files[0], mvAudioFile.files[0],
                    pvAudioFile.files[0], tvAudioFile.files[0]
                ]);

            } catch (error) {

                console.error(
                    "Murmur Error:",
                    error
                );


                if (heartError) {

                    heartError.textContent =
                        error.message ||
                        "Failed to analyze heart sound.";

                    heartError.classList.remove(
                        "hidden"
                    );
                }

            } finally {

                if (heartLoading) {

                    heartLoading.classList.add(
                        "hidden"
                    );
                }


                // Enable again if 4 files exist
                updateMurmurButton();

            }

        }
    );
}


// =====================================================
// AORTIC STENOSIS ANALYSIS
// =====================================================

if (analyzeASButton) {

    analyzeASButton.addEventListener(
        "click",
        async function () {


            if (
                !asAudioFile ||
                !asAudioFile.files.length
            ) {

                return;
            }


            // Disable
            analyzeASButton.disabled = true;


            // Loading
            if (asLoading) {

                asLoading.classList.remove(
                    "hidden"
                );
            }


            // Hide old result
            if (asResult) {

                asResult.classList.add(
                    "hidden"
                );
            }


            // Hide error
            if (asError) {

                asError.classList.add(
                    "hidden"
                );
            }


            try {

                const formData =
                    new FormData();


                formData.append(
                    "file",
                    asAudioFile.files[0]
                );


                const response =
                    await fetch(
                        `${API_URL}/predict-as`,
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await getJSONResponse(
                        response
                    );


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        data.message ||
                        `Server error: ${response.status}`
                    );
                }


                await openResultPage("Aortic Stenosis (AS)", data, [asAudioFile.files[0]]);

            } catch (error) {

                console.error(
                    "AS Error:",
                    error
                );


                if (asError) {

                    asError.textContent =
                        error.message ||
                        "Failed to analyze Aortic Stenosis.";

                    asError.classList.remove(
                        "hidden"
                    );
                }


            } finally {

                if (asLoading) {

                    asLoading.classList.add(
                        "hidden"
                    );
                }


                // Enable again
                if (
                    asAudioFile &&
                    asAudioFile.files.length
                ) {

                    analyzeASButton.disabled =
                        false;
                }

            }

        }
    );
}


// =====================================================
// MITRAL REGURGITATION ANALYSIS
// =====================================================

if (analyzeMRButton) {

    analyzeMRButton.addEventListener(
        "click",
        async function () {


            if (
                !mrAudioFile ||
                !mrAudioFile.files.length
            ) {

                return;
            }


            // Disable
            analyzeMRButton.disabled = true;


            // Loading
            if (mrLoading) {

                mrLoading.classList.remove(
                    "hidden"
                );
            }


            // Hide old result
            if (mrResult) {

                mrResult.classList.add(
                    "hidden"
                );
            }


            // Hide error
            if (mrError) {

                mrError.classList.add(
                    "hidden"
                );
            }


            try {

                const formData =
                    new FormData();


                formData.append(
                    "file",
                    mrAudioFile.files[0]
                );


                const response =
                    await fetch(
                        `${API_URL}/predict-mr`,
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await getJSONResponse(
                        response
                    );


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        data.message ||
                        `Server error: ${response.status}`
                    );
                }


                await openResultPage("Mitral Regurgitation (MR)", data, [mrAudioFile.files[0]]);

            } catch (error) {

                console.error(
                    "MR Error:",
                    error
                );


                if (mrError) {

                    mrError.textContent =
                        error.message ||
                        "Failed to analyze Mitral Regurgitation.";

                    mrError.classList.remove(
                        "hidden"
                    );
                }


            } finally {

                if (mrLoading) {

                    mrLoading.classList.add(
                        "hidden"
                    );
                }


                // Enable again
                if (
                    mrAudioFile &&
                    mrAudioFile.files.length
                ) {

                    analyzeMRButton.disabled =
                        false;
                }

            }

        }
    );
}


// =====================================================
// BACKEND HEALTH CHECK
// =====================================================

async function checkBackend() {

    try {

        const response =
            await fetch(
                `${API_URL}/health`
            );


        const data =
            await response.json();


        if (
            response.ok &&
            data.status === "healthy"
        ) {


            // Server Online
            if (serverStatus) {

                serverStatus.textContent =
                    "Server Online";
            }


            if (serverDot) {

                serverDot.classList.add(
                    "online"
                );

                serverDot.classList.remove(
                    "offline"
                );
            }


        } else {

            throw new Error(
                "Backend unhealthy"
            );
        }


    } catch (error) {

        console.error(
            "Backend connection error:",
            error
        );


        // Server Offline
        if (serverStatus) {

            serverStatus.textContent =
                "Server Offline";
        }


        if (serverDot) {

            serverDot.classList.add(
                "offline"
            );

            serverDot.classList.remove(
                "online"
            );
        }

    }
}


// =====================================================
// START
// =====================================================

checkBackend();


console.log(
    "CardioPulmoCare frontend loaded successfully."
);
