function ProgOlivaTrackOggetti()

  % 1. Carico parametri e creo utilities
  param = getDefaultParameters();
  utilities = createUtilities(param);

  % 2. Inizializzazione variabili
  trackedLocation = [];
  kf = []; % struct del Kalman Filter (inizializzato alla prima detection)
  idxFrame = 0;

  % 3. Loop sui frame
  while hasFrame(utilities.videoReader)
    frame = readFrame(utilities.videoReader);
    idxFrame = idxFrame + 1;

    % Rilevamento 
    [detectedLocation, isObjectDetected, utilities] = detectBall(utilities, frame);

    % Se KF non esiste ancora e abbiamo una detection valida, inizializzalo
    if isempty(kf) && isObjectDetected
      initialLocation = computeInitialLocation(param, detectedLocation);
      kf = initializeKalmanFilter(param, initialLocation);
      trackedLocation = initialLocation;
    end

    % Predizione e correzione se KF esiste
    if ~isempty(kf)
      % Predizione
      kf = predictKalmanFilter(kf);

      % Correzione con la misura, se disponibile
      if isObjectDetected
        kf = correctKalmanFilter(kf, detectedLocation);
      end

      % Aggiorno la posizione stimata
      trackedLocation = [kf.x(1), kf.x(2)];
    end

    % Annotazione e visualizzazione
    annotateTrackedBall(utilities, frame, trackedLocation, idxFrame);

    % Accumulo i risultati per mostrare la traiettoria a fine video
    utilities = accumulateResults(utilities, frame, detectedLocation, trackedLocation);
  end

  % 4. Mostriamo il percorso reale e stimato
  showTrajectory(utilities);
end

%% ===================================================================== %%
%                       F U N Z I O N I   L O C A L I                     %
%% ===================================================================== %%

function param = getDefaultParameters()
  %GETDEFAULTPARAMETERS Parametri di default per il tracking
  param.motionModel           = 'ConstantAcceleration'; % Indicativo
  param.initialLocation       = 'Same as first detection';
  param.initialEstimateError  = 1E5 * ones(1, 3);
  param.motionNoise           = [25, 10, 1];
  param.measurementNoise      = 25;
  param.segmentationThreshold = 0.05;
end

function utilities = createUtilities(param)
  %CREATEUTILITIES Crea gli oggetti per la lettura video e la segmentazione
  utilities.videoReader = VideoReader('/Users/mattiacastiello/Downloads/fileauto4_2.mov')
  %utilities.videoReader = VideoReader('/Users/mattiacastiello/Downloads/IMG_3773_2.mov');
  %utilities.videoReader = VideoReader('/Users/mattiacastiello/Downloads/Registrazione schermo 2024-12-30 alle 15.59.29.mov');

  utilities.videoPlayer = vision.VideoPlayer('Position', [100, 100, 500, 400]);

  utilities.foregroundDetector = vision.ForegroundDetector(...
    'NumTrainingFrames', 10000, ...      % Più frame per imparare lo sfondo
    'InitialVariance', 0.01, ...      % Rendi la segmentazione più o meno sensibile
    'LearningRate', 0.1);           % Tasso di apprendimento dello sfondo

  utilities.blobAnalyzer = vision.BlobAnalysis(...
      'AreaOutputPort', false, ...
      'MinimumBlobArea', 70, ...
      'CentroidOutputPort', true);

  utilities.foregroundMask = [];
  utilities.accumulatedImage      = 0;
  utilities.accumulatedDetections = zeros(0, 2);
  utilities.accumulatedTrackings  = zeros(0, 2);
end

function [detection, isObjectDetected, utilities] = detectBall(utilities, frame)
  %DETECT(foreground + blob analysis)
  grayImage = im2gray(im2single(frame));
  utilities.foregroundMask = step(utilities.foregroundDetector, grayImage);
  detection = step(utilities.blobAnalyzer, utilities.foregroundMask);

  if isempty(detection)
    isObjectDetected = false;
  else
    % Prendiamo solo il primo blob
    detection = detection(1, :);  % [x, y]
    isObjectDetected = true;
  end
figure(1), imshow(utilities.foregroundMask), drawnow

end

function loc = computeInitialLocation(param, detectedLocation)
  %COMPUTEINITIALLOCATION Determina la locazione iniziale (x, y)
  if strcmp(param.initialLocation, 'Same as first detection')
    loc = detectedLocation;
  else
    loc = param.initialLocation; 
  end
end

function kf = initializeKalmanFilter(param, initialLocation)
  %INITIALIZEKALMANFILTER Crea la struct KF: x, P, Q, R, F, H (modello velocità costante)
  kf.x = [initialLocation(1); initialLocation(2); 0; 0];  % [x; y; vx; vy]

  kf.P = diag([param.initialEstimateError(1), ...
               param.initialEstimateError(1), ...
               param.initialEstimateError(2), ...
               param.initialEstimateError(2)]);

  dt = 1; % semplifichiamo
  kf.F = [1  0  dt  0;
          0  1  0   dt;
          0  0  1   0;
          0  0  0   1];

  kf.H = [1 0 0 0;
          0 1 0 0];

  posNoise = param.motionNoise(1);
  velNoise = param.motionNoise(2);
  kf.Q = diag([posNoise, posNoise, velNoise, velNoise]);

  kf.R = param.measurementNoise * eye(2);
end

function kf = predictKalmanFilter(kf)
  %PREDICTKALMANFILTER Fase di predizione
  kf.x = kf.F * kf.x;
  kf.P = kf.F * kf.P * kf.F' + kf.Q;
end

function kf = correctKalmanFilter(kf, measurement)
  %CORRECTKALMANFILTER Fase di correzione
  z = measurement(:);        % [x; y]
  y = z - kf.H * kf.x;       % Innovazione
  S = kf.H * kf.P * kf.H' + kf.R;
  K = kf.P * kf.H' / S;

  kf.x = kf.x + K * y;
  I = eye(size(kf.P));
  kf.P = (I - K * kf.H) * kf.P;
end

function annotateTrackedBall(utilities, frame, trackedLocation, idxFrame)
  %ANNOTATETRACKEDBALL Visualizzazione del frame con posizione stimata
  combinedImage = max(repmat(utilities.foregroundMask, [1,1,3]), im2single(frame));
  label = sprintf('Frame %d', idxFrame);

  if ~isempty(trackedLocation)
    shape = 'circle';
    region = [trackedLocation(1), trackedLocation(2), 5]; % [x, y, r=5]
    combinedImage = insertObjectAnnotation(combinedImage, shape, region, label, ...
      'Color', 'red', 'LineWidth', 2);
  end

  step(utilities.videoPlayer, combinedImage);

  % Esempio: visualizzare un frame speciale
  if idxFrame == 40
    figure;
    imshow(combinedImage);
    title('Frame 40 - Dettaglio');
  end
end

function utilities = accumulateResults(utilities, frame, detectedLocation, trackedLocation)
  %ACCUMULATERESULTS Salva frame e posizioni di detection e tracking
  utilities.accumulatedImage = max(utilities.accumulatedImage, frame);

  if ~isempty(detectedLocation)
    utilities.accumulatedDetections = ...
      [utilities.accumulatedDetections; detectedLocation];
  end

  if ~isempty(trackedLocation)
    utilities.accumulatedTrackings = ...
      [utilities.accumulatedTrackings; trackedLocation];
  end
end

function showTrajectory(utilities)
  %SHOWTRAJECTORY Mostra il percorso reale vs. stimato
  uiscopes.close('All');

  figure;
  imshow(utilities.accumulatedImage/2 + 0.5); hold on;
  title('Percorso reale (detection) e percorso stimato (KF)');

  if ~isempty(utilities.accumulatedDetections)
    plot(utilities.accumulatedDetections(:,1), ...
         utilities.accumulatedDetections(:,2), 'k+');
  end

  if ~isempty(utilities.accumulatedTrackings)
    plot(utilities.accumulatedTrackings(:,1), ...
         utilities.accumulatedTrackings(:,2), 'r-o');
    legend('Percorso Reale', 'Percorso Stimato');
  end
end
