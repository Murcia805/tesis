import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import {HomeComponent} from './home/home.component';
import {ContactoComponent} from './contacto/contacto.component';

const routes: Routes = [
  {path: 'encuesta', component: HomeComponent},
  {path: 'contacto', component: ContactoComponent},
  { path: '', redirectTo: '/encuesta', pathMatch: 'full' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
