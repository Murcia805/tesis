import {ChangeDetectorRef, Component, OnInit} from '@angular/core';
import WaveSurfer from 'wavesurfer.js';
import TimelinePlugin from 'wavesurfer.js/dist/plugin/wavesurfer.timeline.min.js';
import Regions from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js';
import {ApiService} from '../services/api.service';


@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})

export class HomeComponent implements OnInit {

  private selectedFiles: FileList;
  public selectedModel: number = 1;

  public wave: WaveSurfer = null;
  public graph = undefined;
  public isPausado = true;

  public isTerminoEncuesta = false;

  private arreglo = [];
  public bpm = 0;
  private index = 0;

  private infoCancion: any;

  constructor(private cdr: ChangeDetectorRef,
              private service: ApiService) {
  }

  ngOnInit() {
  }

  async onPreviewPressed() {
    this.wave = null;
    await this.service.getMusica().toPromise().then(
      (response: any) => {
        this.infoCancion = response;
        this.generateWaveform(response.index, response.tiempo);
        this.cdr.detectChanges();
      }
    );
  }

  controlReproduccion(): void {
    if (this.isPausado) {
      this.isPausado = false;
      this.wave.play();
    } else {
      this.isPausado = true;
      this.wave.stop();
    }

  }

  marcar() {
    const time = this.wave.getCurrentTime();
    this.arreglo.push(time);
    this.wave.regions.add({
      start: time,
      end: time + 0.03,
      resize: false,
      drag: false,
      color: 'rgba(255,0,0, 0.7)'
    });
    console.log(this.wave.regions);
  }

  limpiarRegiones() {
    this.wave.regions.clear();
    this.arreglo = [];
  }

  CalculoBPM(): number {
    this.bpm += this.arreglo[this.index];
    return 60000 / this.bpm;
  }

  async generateWaveform(indexCancion: number, tiempoInicio: number) {
    if (this.wave) {
      const videoElement = document.getElementById('waveform');
      videoElement.removeAttribute('src');
    }
    this.wave = await WaveSurfer.create({
      container: '#waveform',
      waveColor: 'blue',
      progressColor: 'black',
      plugins: [
        TimelinePlugin.create({
          container: '#wave-timeline',
          notchPercentHeight: 80,
          fontFamily: 'Helvetica Neue'
        }),
        Regions.create()
      ]
    });

    this.service.postMusica(indexCancion, tiempoInicio).subscribe(
      (response: any) => {
        const file = new Blob([response], { type: 'audio/mpeg' });
        this.wave.loadBlob(file);
        this.getArregloBeats(indexCancion);
      }
    );

    this.wave.on('ready', () => {
      alert('Cargado Exitosamente');
    });
  }

  subirEncuesta() {
    const dato = {
      index: this.infoCancion.index,
      nombre: this.infoCancion.nombre,
      duracion: this.infoCancion.duracion,
      tiempo: this.infoCancion.tiempo,
      marcas: this.arreglo
    };

    this.service.postInsertarEncuesta(dato).toPromise().then(
      (respuesta: any) => {
        console.log(respuesta);
        this.isTerminoEncuesta = true;
      }
    );
  }

  getArregloBeats(index: number) {
    this.service.postArrayBeats(index).toPromise().then(
      (respuesta: number[]) => {
        // tslint:disable-next-line:prefer-for-of
        for (let i = 0; i < respuesta.length; i++) {
          this.wave.regions.add({
            start: respuesta[i],
            end: respuesta[i] + 0.03,
            resize: false,
            drag: false,
            color: 'rgba(255,0,0, 0.7)'
          });
        }
      }
    );
  }

  reHacerEncuesta() {
    this.wave = null;
    this.graph = undefined;
    this.isPausado = true;

    this.isTerminoEncuesta = false;
    this.arreglo = [];
    this.bpm = 0;
    this.index = 0;

    this.infoCancion = null;
  }

  selectFile(event) {
    this.selectedFiles = event.target.files;
  }

  uploadFile() {
    const file = this.selectedFiles.item(0);
    const formData: FormData = new FormData();
    formData.append('file', file, file.name);
    this.service.postPrediccion(this.selectedModel, formData).toPromise().then(
      (respuesta: any) => {
        alert('BPM = ' + respuesta);
      }
    );
  }
}
