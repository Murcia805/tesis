import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {environment} from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private MAIN_URL = '';
  private HTTP_HEADER: HttpHeaders;

  constructor(private http: HttpClient) {
    this.MAIN_URL = environment.MUSICA_API;
  }

  public getMusica() {
    return this.http.get(
      this.MAIN_URL
    );
  }

  public postMusica(indexCancion: number, tiempoInicio: number) {
    const data = {
      numero: indexCancion,
      tiempo: tiempoInicio
    };
    return this.http.post(
      this.MAIN_URL,
      data,
      { responseType: 'blob' }
    );
  }

  public postInsertarEncuesta(data: any) {
    const URL = `${this.MAIN_URL}/guardar`;
    return this.http.post(
      URL,
      data,
      { headers: this.HTTP_HEADER }
    );
  }

  public postArrayBeats(indexCancion: number) {
    const data = {
      numero: indexCancion,
    };
    const URL = `${this.MAIN_URL}/obtenerArregloBeats`;
    return this.http.post(
      URL,
      data,
      { headers: this.HTTP_HEADER }
    );
  }

  public postPrediccion(modelo: number, data: any) {
    const URL = `${this.MAIN_URL}/prediccion?modelo=${modelo}`;
    this.HTTP_HEADER = new HttpHeaders().set('Accept', '*/*');

    return this.http.post(
      URL,
      data,
      { headers: this.HTTP_HEADER }
    );


  }
}
